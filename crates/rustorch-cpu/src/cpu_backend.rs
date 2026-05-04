//! `CpuBackend` — the v1 default backend.
//!
//! Implements the [`Backend`](crate::backend::Backend) trait surface
//! for f32 + i32 + i64 element types, dispatching through
//! [`crate::iterator::map_unary_same`] / [`map_binary_same`] /
//! [`map_binary`].
//!
//! Used by `rustorch::cpu::cpu_backend()` to obtain the static
//! singleton (registered via `OnceLock`).

use crate::backend::Backend;
use crate::error::BackendError;
use crate::iterator::{map_binary, map_binary_same, map_unary_same};
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

/// CPU backend singleton. Stateless — every operation is a kernel
/// dispatch over the input tensors.
#[derive(Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Construct a new CPU backend handle. Cheap (no allocation).
    pub const fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        dispatch_binary(lhs, rhs, "add", BinaryKind::Add)
    }

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        dispatch_binary(lhs, rhs, "sub", BinaryKind::Sub)
    }

    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        dispatch_binary(lhs, rhs, "mul", BinaryKind::Mul)
    }

    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        dispatch_binary(lhs, rhs, "div", BinaryKind::Div)
    }

    fn neg(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        match src.dtype() {
            Dtype::F32 => map_unary_same::<f32, _>(src, "neg", |x| -x),
            Dtype::F64 => map_unary_same::<f64, _>(src, "neg", |x| -x),
            Dtype::I64 => map_unary_same::<i64, _>(src, "neg", |x| -x),
            Dtype::I32 => map_unary_same::<i32, _>(src, "neg", |x| -x),
            Dtype::I8 => map_unary_same::<i8, _>(src, "neg", |x| x.wrapping_neg()),
            d => Err(BackendError::UnsupportedOp {
                op: "neg",
                device: self.name(),
            })
            .map_err(|_| BackendError::DtypeMismatch {
                op: "neg",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        if lhs.dtype() != rhs.dtype() {
            return Err(BackendError::DtypeMismatch {
                op: "matmul",
                lhs: lhs.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if lhs.ndim() != 2 || rhs.ndim() != 2 {
            return Err(BackendError::ShapeMismatch {
                op: "matmul",
                lhs: lhs.shape().to_vec(),
                rhs: rhs.shape().to_vec(),
            });
        }
        let (m, k1) = (lhs.shape()[0], lhs.shape()[1]);
        let (k2, n) = (rhs.shape()[0], rhs.shape()[1]);
        if k1 != k2 {
            return Err(BackendError::ShapeMismatch {
                op: "matmul",
                lhs: lhs.shape().to_vec(),
                rhs: rhs.shape().to_vec(),
            });
        }
        match lhs.dtype() {
            Dtype::F32 => matmul_dispatch_f32(lhs, rhs, m, k1, n),
            Dtype::F64 => matmul_dispatch_f64(lhs, rhs, m, k1, n),
            // bf16/f16 do not implement Add/Mul natively, so we accumulate
            // through f32 and cast the final result back. Matches the
            // numerics of GPU bf16 matmul (which accumulates in f32).
            Dtype::BF16 => {
                let lhs_f = lhs.to_dtype(Dtype::F32);
                let rhs_f = rhs.to_dtype(Dtype::F32);
                let out_f = matmul_dispatch_f32(&lhs_f, &rhs_f, m, k1, n)?;
                Ok(out_f.to_dtype(Dtype::BF16))
            },
            Dtype::F16 => {
                let lhs_f = lhs.to_dtype(Dtype::F32);
                let rhs_f = rhs.to_dtype(Dtype::F32);
                let out_f = matmul_dispatch_f32(&lhs_f, &rhs_f, m, k1, n)?;
                Ok(out_f.to_dtype(Dtype::F16))
            },
            d => Err(BackendError::DtypeMismatch {
                op: "matmul",
                lhs: d,
                rhs: d,
            }),
        }
    }

    /// Transpose-aware matmul fast path. On macOS we forward the
    /// `transpose_a` / `transpose_b` flags directly to `cblas_sgemm`'s
    /// `transa` / `transb` arguments, skipping the materialised
    /// transpose buffer + the second matmul pass that the default
    /// trait impl would emit. In `MatMulBackward` this saves two
    /// 4 MB transposes + their re-materialisation as Vec<f32> per
    /// training step.
    ///
    /// Falls back to the default `transpose + matmul` for non-rank-2,
    /// non-f32, or wasm32 builds (where Accelerate isn't linked).
    #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
    fn matmul_with_transposes(
        &self,
        lhs: &Tensor,
        rhs: &Tensor,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Tensor, BackendError> {
        // Only the f32 + rank-2 path goes through Accelerate — other
        // dtypes fall back to the default `transpose + matmul` impl.
        if lhs.dtype() != Dtype::F32
            || rhs.dtype() != Dtype::F32
            || lhs.ndim() != 2
            || rhs.ndim() != 2
        {
            // Default impl: transpose + matmul.
            let lhs_eff = if transpose_a {
                self.transpose(lhs, 0, 1)?
            } else {
                lhs.clone()
            };
            let rhs_eff = if transpose_b {
                self.transpose(rhs, 0, 1)?
            } else {
                rhs.clone()
            };
            return self.matmul(&lhs_eff, &rhs_eff);
        }
        let l_shape = lhs.shape();
        let r_shape = rhs.shape();
        // Effective dimensions after the implicit transpose:
        //   transpose_a:  A is [K, M], output rows = M = A.shape[1]
        //   transpose_b:  B is [N, K], output cols = N = B.shape[0]
        let (m, k_lhs) = if transpose_a {
            (l_shape[1], l_shape[0])
        } else {
            (l_shape[0], l_shape[1])
        };
        let (k_rhs, n) = if transpose_b {
            (r_shape[1], r_shape[0])
        } else {
            (r_shape[0], r_shape[1])
        };
        if k_lhs != k_rhs {
            return Err(BackendError::ShapeMismatch {
                op: "matmul_with_transposes",
                lhs: l_shape.to_vec(),
                rhs: r_shape.to_vec(),
            });
        }
        let k = k_lhs;
        if m < GEMM_DISPATCH_MIN || n < GEMM_DISPATCH_MIN || k < GEMM_DISPATCH_MIN {
            // Tiny shapes: fall back to default (transpose + matmul) so
            // the scalar `matmul_naive` path stays consistent.
            let lhs_eff = if transpose_a {
                self.transpose(lhs, 0, 1)?
            } else {
                lhs.clone()
            };
            let rhs_eff = if transpose_b {
                self.transpose(rhs, 0, 1)?
            } else {
                rhs.clone()
            };
            return self.matmul(&lhs_eff, &rhs_eff);
        }

        let lhs_slice = lhs.as_slice::<f32>().ok_or(BackendError::DtypeMismatch {
            op: "matmul_with_transposes",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        })?;
        let rhs_slice = rhs.as_slice::<f32>().ok_or(BackendError::DtypeMismatch {
            op: "matmul_with_transposes",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        })?;
        let mut out_buf: Vec<f32> = vec![0.0_f32; m * n];

        // SAFETY: input slices are exactly the right size for their
        // (un-transposed) shape; out_buf is m*n; transpose flags are
        // honoured via the cblas transa/transb arguments.
        unsafe {
            crate::accelerate::cblas_sgemm(
                crate::accelerate::CBLAS_ROW_MAJOR,
                if transpose_a {
                    crate::accelerate::CBLAS_TRANS
                } else {
                    crate::accelerate::CBLAS_NO_TRANS
                },
                if transpose_b {
                    crate::accelerate::CBLAS_TRANS
                } else {
                    crate::accelerate::CBLAS_NO_TRANS
                },
                m as std::os::raw::c_int,
                n as std::os::raw::c_int,
                k as std::os::raw::c_int,
                1.0,
                lhs_slice.as_ptr(),
                // lda is the leading dim of A in storage order (un-transposed):
                //   transpose_a == false: A is [m, k] row-major → lda = k
                //   transpose_a == true:  A is [k, m] row-major → lda = m
                if transpose_a { m } else { k } as std::os::raw::c_int,
                rhs_slice.as_ptr(),
                if transpose_b { k } else { n } as std::os::raw::c_int,
                0.0,
                out_buf.as_mut_ptr(),
                n as std::os::raw::c_int,
            );
        }
        Tensor::from_vec_typed::<f32, _>(vec![m, n], out_buf).map_err(|_| {
            BackendError::OutOfMemory {
                bytes: m * n * core::mem::size_of::<f32>(),
            }
        })
    }

    /// CPU `linear` forward: `C = A @ B + bias_broadcast(N)` in a single
    /// `cblas_sgemm` call followed by an in-place parallel bias add.
    /// Skips the separate `add_bias` dispatch that the default trait
    /// impl would emit (which itself allocates a `[M, N]` intermediate
    /// before returning a fresh tensor).
    #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
    fn matmul_with_bias(
        &self,
        lhs: &Tensor,
        rhs: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor, BackendError> {
        // Fast-path eligibility: rank-2 f32 + rank-1 f32 bias matching N.
        if lhs.dtype() != Dtype::F32
            || rhs.dtype() != Dtype::F32
            || bias.dtype() != Dtype::F32
            || lhs.ndim() != 2
            || rhs.ndim() != 2
            || bias.ndim() != 1
        {
            // Default composition (matmul + add_bias).
            let mm = self.matmul(lhs, rhs)?;
            return self.add_bias(&mm, bias);
        }
        let l_shape = lhs.shape();
        let r_shape = rhs.shape();
        let bias_shape = bias.shape();
        let (m, k1) = (l_shape[0], l_shape[1]);
        let (k2, n) = (r_shape[0], r_shape[1]);
        if k1 != k2 || bias_shape[0] != n {
            // Shape error — let the default path surface it cleanly.
            let mm = self.matmul(lhs, rhs)?;
            return self.add_bias(&mm, bias);
        }
        if m < GEMM_DISPATCH_MIN || n < GEMM_DISPATCH_MIN || k1 < GEMM_DISPATCH_MIN {
            let mm = self.matmul(lhs, rhs)?;
            return self.add_bias(&mm, bias);
        }
        // Materialise contiguous f32 inputs (zero-copy when already
        // contiguous).
        let lhs_owned: Option<Vec<f32>>;
        let rhs_owned: Option<Vec<f32>>;
        let lhs_slice: &[f32] = if let Some(s) = lhs.as_slice::<f32>() {
            lhs_owned = None;
            s
        } else {
            lhs_owned = Some(
                lhs.iter_elements::<f32>()
                    .ok_or(BackendError::DtypeMismatch {
                        op: "matmul_with_bias",
                        lhs: lhs.dtype(),
                        rhs: rhs.dtype(),
                    })?
                    .collect(),
            );
            lhs_owned.as_deref().unwrap()
        };
        let rhs_slice: &[f32] = if let Some(s) = rhs.as_slice::<f32>() {
            rhs_owned = None;
            s
        } else {
            rhs_owned = Some(
                rhs.iter_elements::<f32>()
                    .ok_or(BackendError::DtypeMismatch {
                        op: "matmul_with_bias",
                        lhs: lhs.dtype(),
                        rhs: rhs.dtype(),
                    })?
                    .collect(),
            );
            rhs_owned.as_deref().unwrap()
        };
        let bias_slice: &[f32] = bias.as_slice::<f32>().ok_or(BackendError::DtypeMismatch {
            op: "matmul_with_bias",
            lhs: lhs.dtype(),
            rhs: bias.dtype(),
        })?;

        let mut out_buf: Vec<f32> = vec![0.0_f32; m * n];

        // SAFETY: buffers sized correctly (m·k, k·n, m·n).
        unsafe {
            crate::accelerate::sgemm_row_major(m, k1, n, lhs_slice, rhs_slice, &mut out_buf);
        }
        drop(lhs_owned);
        drop(rhs_owned);

        // Parallel in-place bias broadcast: out[b, j] += bias[j].
        use rayon::prelude::*;
        const PARALLEL_THRESHOLD: usize = 16_384;
        if m * n < PARALLEL_THRESHOLD {
            for b in 0..m {
                let row = &mut out_buf[b * n..b * n + n];
                for j in 0..n {
                    row[j] += bias_slice[j];
                }
            }
        } else {
            out_buf.par_chunks_mut(n).for_each(|row| {
                for j in 0..n {
                    row[j] += bias_slice[j];
                }
            });
        }

        Tensor::from_vec_typed::<f32, _>(vec![m, n], out_buf).map_err(|_| {
            BackendError::OutOfMemory {
                bytes: m * n * core::mem::size_of::<f32>(),
            }
        })
    }

    fn sum(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        match src.dtype() {
            Dtype::F32 => sum_kernel::<f32>(src, 0.0_f32, |a, b| a + b),
            Dtype::F64 => sum_kernel::<f64>(src, 0.0_f64, |a, b| a + b),
            Dtype::I64 => sum_kernel::<i64>(src, 0_i64, |a, b| a + b),
            Dtype::I32 => sum_kernel::<i32>(src, 0_i32, |a, b| a + b),
            d => Err(BackendError::DtypeMismatch {
                op: "sum",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn mean(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        let n = src.numel();
        if n == 0 {
            return Err(BackendError::NumericalError(
                "mean of empty tensor is NaN — refused".into(),
            ));
        }
        match src.dtype() {
            Dtype::F32 => sum_kernel::<f32>(src, 0.0_f32, |a, b| a + b)
                .map(|t| Tensor::scalar(t.as_slice::<f32>().unwrap()[0] / n as f32)),
            Dtype::F64 => {
                let s = sum_kernel::<f64>(src, 0.0_f64, |a, b| a + b)?;
                let s_val = s.as_slice::<f64>().unwrap()[0];
                Tensor::from_vec_typed::<f64, _>([], vec![s_val / n as f64]).map_err(|_| {
                    BackendError::OutOfMemory {
                        bytes: core::mem::size_of::<f64>(),
                    }
                })
            },
            d => Err(BackendError::DtypeMismatch {
                op: "mean",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn relu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        match src.dtype() {
            Dtype::F32 => map_unary_same::<f32, _>(src, "relu", |x| x.max(0.0)),
            Dtype::F64 => map_unary_same::<f64, _>(src, "relu", |x| x.max(0.0)),
            d => Err(BackendError::DtypeMismatch {
                op: "relu",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn eq(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cmp(lhs, rhs, "eq", CmpKind::Eq)
    }

    fn ne(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cmp(lhs, rhs, "ne", CmpKind::Ne)
    }

    fn lt(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cmp(lhs, rhs, "lt", CmpKind::Lt)
    }

    fn le(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cmp(lhs, rhs, "le", CmpKind::Le)
    }

    fn gt(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cmp(lhs, rhs, "gt", CmpKind::Gt)
    }

    fn ge(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cmp(lhs, rhs, "ge", CmpKind::Ge)
    }

    fn abs(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        match src.dtype() {
            Dtype::F32 => map_unary_same::<f32, _>(src, "abs", |x| x.abs()),
            Dtype::F64 => map_unary_same::<f64, _>(src, "abs", |x| x.abs()),
            Dtype::I64 => map_unary_same::<i64, _>(src, "abs", |x| x.wrapping_abs()),
            Dtype::I32 => map_unary_same::<i32, _>(src, "abs", |x| x.wrapping_abs()),
            d => Err(BackendError::DtypeMismatch {
                op: "abs",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn sqrt(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "sqrt", f32::sqrt, f64::sqrt)
    }

    fn exp(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "exp", f32::exp, f64::exp)
    }

    fn log(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "log", f32::ln, f64::ln)
    }

    fn sin(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "sin", f32::sin, f64::sin)
    }

    fn cos(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "cos", f32::cos, f64::cos)
    }

    fn tan(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "tan", f32::tan, f64::tan)
    }

    fn pow_scalar(&self, src: &Tensor, exponent: f64) -> Result<Tensor, BackendError> {
        match src.dtype() {
            Dtype::F32 => {
                let e = exponent as f32;
                map_unary_same::<f32, _>(src, "pow_scalar", move |x| x.powf(e))
            },
            Dtype::F64 => map_unary_same::<f64, _>(src, "pow_scalar", move |x| x.powf(exponent)),
            d => Err(BackendError::DtypeMismatch {
                op: "pow_scalar",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn sigmoid(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(
            src,
            "sigmoid",
            |x| 1.0 / (1.0 + (-x).exp()),
            |x| 1.0 / (1.0 + (-x).exp()),
        )
    }

    fn tanh(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "tanh", f32::tanh, f64::tanh)
    }

    fn gelu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        // Approximation tanh — matches PyTorch's `gelu(approximate='tanh')`
        // which is faster + numerically stable. Exact gelu via erf can be
        // wired in once we have an erf kernel.
        const C: f32 = 0.797_884_5_f32; // sqrt(2/pi)
        const C64: f64 = 0.797_884_560_802_865_f64;
        unary_float(
            src,
            "gelu",
            |x| 0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh()),
            |x| 0.5 * x * (1.0 + (C64 * (x + 0.044_715 * x * x * x)).tanh()),
        )
    }

    fn leaky_relu(&self, src: &Tensor, slope: f64) -> Result<Tensor, BackendError> {
        match src.dtype() {
            Dtype::F32 => {
                let s = slope as f32;
                map_unary_same::<f32, _>(
                    src,
                    "leaky_relu",
                    move |x| if x > 0.0 { x } else { s * x },
                )
            },
            Dtype::F64 => {
                map_unary_same::<f64, _>(
                    src,
                    "leaky_relu",
                    move |x| {
                        if x > 0.0 {
                            x
                        } else {
                            slope * x
                        }
                    },
                )
            },
            d => Err(BackendError::DtypeMismatch {
                op: "leaky_relu",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn silu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(
            src,
            "silu",
            |x| x / (1.0 + (-x).exp()),
            |x| x / (1.0 + (-x).exp()),
        )
    }

    fn isnan(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        match src.dtype() {
            Dtype::F32 => crate::iterator::map_unary::<f32, bool, _>(src, "isnan", |x| x.is_nan()),
            Dtype::F64 => crate::iterator::map_unary::<f64, bool, _>(src, "isnan", |x| x.is_nan()),
            d => Err(BackendError::DtypeMismatch {
                op: "isnan",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn isinf(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        match src.dtype() {
            Dtype::F32 => {
                crate::iterator::map_unary::<f32, bool, _>(src, "isinf", |x| x.is_infinite())
            },
            Dtype::F64 => {
                crate::iterator::map_unary::<f64, bool, _>(src, "isinf", |x| x.is_infinite())
            },
            d => Err(BackendError::DtypeMismatch {
                op: "isinf",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn isfinite(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        match src.dtype() {
            Dtype::F32 => {
                crate::iterator::map_unary::<f32, bool, _>(src, "isfinite", |x| x.is_finite())
            },
            Dtype::F64 => {
                crate::iterator::map_unary::<f64, bool, _>(src, "isfinite", |x| x.is_finite())
            },
            d => Err(BackendError::DtypeMismatch {
                op: "isfinite",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn pow(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        if lhs.dtype() != rhs.dtype() {
            return Err(BackendError::DtypeMismatch {
                op: "pow",
                lhs: lhs.dtype(),
                rhs: rhs.dtype(),
            });
        }
        match lhs.dtype() {
            Dtype::F32 => map_binary_same::<f32, _>(lhs, rhs, "pow", |a, b| a.powf(b)),
            Dtype::F64 => map_binary_same::<f64, _>(lhs, rhs, "pow", |a, b| a.powf(b)),
            d => Err(BackendError::DtypeMismatch {
                op: "pow",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn rsqrt(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "rsqrt", |x| 1.0 / x.sqrt(), |x| 1.0 / x.sqrt())
    }

    fn expm1(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "expm1", f32::exp_m1, f64::exp_m1)
    }

    fn log1p(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "log1p", f32::ln_1p, f64::ln_1p)
    }

    fn log2(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "log2", f32::log2, f64::log2)
    }

    fn log10(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "log10", f32::log10, f64::log10)
    }

    fn asin(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "asin", f32::asin, f64::asin)
    }

    fn acos(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "acos", f32::acos, f64::acos)
    }

    fn atan(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "atan", f32::atan, f64::atan)
    }

    fn atan2(&self, y: &Tensor, x: &Tensor) -> Result<Tensor, BackendError> {
        if y.dtype() != x.dtype() {
            return Err(BackendError::DtypeMismatch {
                op: "atan2",
                lhs: y.dtype(),
                rhs: x.dtype(),
            });
        }
        match y.dtype() {
            Dtype::F32 => map_binary_same::<f32, _>(y, x, "atan2", |a, b| a.atan2(b)),
            Dtype::F64 => map_binary_same::<f64, _>(y, x, "atan2", |a, b| a.atan2(b)),
            d => Err(BackendError::DtypeMismatch {
                op: "atan2",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn sinh(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "sinh", f32::sinh, f64::sinh)
    }

    fn cosh(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(src, "cosh", f32::cosh, f64::cosh)
    }

    fn elu(&self, src: &Tensor, alpha: f64) -> Result<Tensor, BackendError> {
        match src.dtype() {
            Dtype::F32 => {
                let a = alpha as f32;
                map_unary_same::<f32, _>(src, "elu", move |x| {
                    if x > 0.0 {
                        x
                    } else {
                        a * ((x).exp() - 1.0)
                    }
                })
            },
            Dtype::F64 => map_unary_same::<f64, _>(src, "elu", move |x| {
                if x > 0.0 {
                    x
                } else {
                    alpha * (x.exp() - 1.0)
                }
            }),
            d => Err(BackendError::DtypeMismatch {
                op: "elu",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn softplus(&self, src: &Tensor, beta: f64) -> Result<Tensor, BackendError> {
        // Numerically stable softplus: when beta*x is large, x ≈ x;
        // when beta*x is small, log(1 + exp(beta*x)).
        match src.dtype() {
            Dtype::F32 => {
                let beta_f = beta as f32;
                map_unary_same::<f32, _>(src, "softplus", move |x| {
                    let bx = beta_f * x;
                    if bx > 20.0 {
                        x
                    } else {
                        (1.0 + bx.exp()).ln() / beta_f
                    }
                })
            },
            Dtype::F64 => map_unary_same::<f64, _>(src, "softplus", move |x| {
                let bx = beta * x;
                if bx > 20.0 {
                    x
                } else {
                    (1.0 + bx.exp()).ln() / beta
                }
            }),
            d => Err(BackendError::DtypeMismatch {
                op: "softplus",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn hardswish(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(
            src,
            "hardswish",
            |x| x * ((x + 3.0).clamp(0.0, 6.0)) / 6.0,
            |x| x * ((x + 3.0).clamp(0.0, 6.0)) / 6.0,
        )
    }

    fn hardtanh(&self, src: &Tensor, min: f64, max: f64) -> Result<Tensor, BackendError> {
        match src.dtype() {
            Dtype::F32 => {
                let lo = min as f32;
                let hi = max as f32;
                map_unary_same::<f32, _>(src, "hardtanh", move |x| x.clamp(lo, hi))
            },
            Dtype::F64 => map_unary_same::<f64, _>(src, "hardtanh", move |x| x.clamp(min, max)),
            d => Err(BackendError::DtypeMismatch {
                op: "hardtanh",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn gelu_exact(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        // gelu_exact(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        // erf via Abramowitz-Stegun 7.1.26 (max abs error ≤ 1.5e-7).
        const SQRT_2: f32 = core::f32::consts::SQRT_2;
        const SQRT_2_F64: f64 = core::f64::consts::SQRT_2;
        unary_float(
            src,
            "gelu_exact",
            |x| 0.5 * x * (1.0 + erf_f32(x / SQRT_2)),
            |x| 0.5 * x * (1.0 + erf_f64(x / SQRT_2_F64)),
        )
    }

    fn hardsigmoid(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_float(
            src,
            "hardsigmoid",
            |x| ((x + 3.0) / 6.0).clamp(0.0, 1.0),
            |x| ((x + 3.0) / 6.0).clamp(0.0, 1.0),
        )
    }

    // -------------------- Indexing kernels --------------------

    fn gather(&self, src: &Tensor, dim: usize, idx: &Tensor) -> Result<Tensor, BackendError> {
        if idx.dtype() != Dtype::I64 {
            return Err(BackendError::DtypeMismatch {
                op: "gather",
                lhs: idx.dtype(),
                rhs: Dtype::I64,
            });
        }
        if src.ndim() != idx.ndim() {
            return Err(BackendError::ShapeMismatch {
                op: "gather",
                lhs: src.shape().to_vec(),
                rhs: idx.shape().to_vec(),
            });
        }
        if dim >= src.ndim() {
            return Err(BackendError::IndexOutOfBounds {
                op: "gather",
                index: dim as i64,
                bound: src.ndim(),
            });
        }
        match src.dtype() {
            Dtype::F32 => gather_typed::<f32>(src, dim, idx),
            Dtype::F64 => gather_typed::<f64>(src, dim, idx),
            Dtype::I64 => gather_typed::<i64>(src, dim, idx),
            Dtype::I32 => gather_typed::<i32>(src, dim, idx),
            d => Err(BackendError::DtypeMismatch {
                op: "gather",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn scatter(
        &self,
        dst: &Tensor,
        dim: usize,
        idx: &Tensor,
        src: &Tensor,
    ) -> Result<Tensor, BackendError> {
        scatter_dispatch(dst, dim, idx, src, "scatter", false)
    }

    fn scatter_add(
        &self,
        dst: &Tensor,
        dim: usize,
        idx: &Tensor,
        src: &Tensor,
    ) -> Result<Tensor, BackendError> {
        scatter_dispatch(dst, dim, idx, src, "scatter_add", true)
    }

    fn index_select(
        &self,
        src: &Tensor,
        dim: usize,
        indices: &Tensor,
    ) -> Result<Tensor, BackendError> {
        if indices.dtype() != Dtype::I64 {
            return Err(BackendError::DtypeMismatch {
                op: "index_select",
                lhs: indices.dtype(),
                rhs: Dtype::I64,
            });
        }
        if indices.ndim() != 1 {
            return Err(BackendError::ShapeMismatch {
                op: "index_select",
                lhs: indices.shape().to_vec(),
                rhs: vec![indices.numel()],
            });
        }
        if dim >= src.ndim() {
            return Err(BackendError::IndexOutOfBounds {
                op: "index_select",
                index: dim as i64,
                bound: src.ndim(),
            });
        }
        match src.dtype() {
            Dtype::F32 => index_select_typed::<f32>(src, dim, indices),
            Dtype::F64 => index_select_typed::<f64>(src, dim, indices),
            Dtype::I64 => index_select_typed::<i64>(src, dim, indices),
            Dtype::I32 => index_select_typed::<i32>(src, dim, indices),
            d => Err(BackendError::DtypeMismatch {
                op: "index_select",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn masked_select(&self, src: &Tensor, mask: &Tensor) -> Result<Tensor, BackendError> {
        if mask.dtype() != Dtype::Bool {
            return Err(BackendError::DtypeMismatch {
                op: "masked_select",
                lhs: mask.dtype(),
                rhs: Dtype::Bool,
            });
        }
        if mask.shape() != src.shape() {
            return Err(BackendError::ShapeMismatch {
                op: "masked_select",
                lhs: src.shape().to_vec(),
                rhs: mask.shape().to_vec(),
            });
        }
        match src.dtype() {
            Dtype::F32 => masked_select_typed::<f32>(src, mask),
            Dtype::F64 => masked_select_typed::<f64>(src, mask),
            Dtype::I64 => masked_select_typed::<i64>(src, mask),
            Dtype::I32 => masked_select_typed::<i32>(src, mask),
            d => Err(BackendError::DtypeMismatch {
                op: "masked_select",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn masked_fill(&self, src: &Tensor, mask: &Tensor, value: f64) -> Result<Tensor, BackendError> {
        if mask.dtype() != Dtype::Bool {
            return Err(BackendError::DtypeMismatch {
                op: "masked_fill",
                lhs: mask.dtype(),
                rhs: Dtype::Bool,
            });
        }
        if mask.shape() != src.shape() {
            return Err(BackendError::ShapeMismatch {
                op: "masked_fill",
                lhs: src.shape().to_vec(),
                rhs: mask.shape().to_vec(),
            });
        }
        match src.dtype() {
            Dtype::F32 => masked_fill_typed::<f32>(src, mask, value as f32),
            Dtype::F64 => masked_fill_typed::<f64>(src, mask, value),
            Dtype::I64 => masked_fill_typed::<i64>(src, mask, value as i64),
            Dtype::I32 => masked_fill_typed::<i32>(src, mask, value as i32),
            d => Err(BackendError::DtypeMismatch {
                op: "masked_fill",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn r#where(&self, cond: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor, BackendError> {
        if cond.dtype() != Dtype::Bool {
            return Err(BackendError::DtypeMismatch {
                op: "where",
                lhs: cond.dtype(),
                rhs: Dtype::Bool,
            });
        }
        if x.dtype() != y.dtype() {
            return Err(BackendError::DtypeMismatch {
                op: "where",
                lhs: x.dtype(),
                rhs: y.dtype(),
            });
        }
        // shapes must broadcast — for v1, require all three to have the same shape.
        if cond.shape() != x.shape() || x.shape() != y.shape() {
            return Err(BackendError::ShapeMismatch {
                op: "where",
                lhs: x.shape().to_vec(),
                rhs: y.shape().to_vec(),
            });
        }
        match x.dtype() {
            Dtype::F32 => where_typed::<f32>(cond, x, y),
            Dtype::F64 => where_typed::<f64>(cond, x, y),
            Dtype::I64 => where_typed::<i64>(cond, x, y),
            Dtype::I32 => where_typed::<i32>(cond, x, y),
            d => Err(BackendError::DtypeMismatch {
                op: "where",
                lhs: d,
                rhs: d,
            }),
        }
    }

    fn nonzero(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        // Build a [N, ndim] I64 tensor where N is the count of nonzero
        // elements in src.
        let ndim = src.ndim();
        let shape = src.shape().to_vec();
        let nonzero_coords: Vec<i64>;
        let walk = |is_nz: &dyn Fn(usize) -> bool| -> Vec<i64> {
            let mut acc = Vec::new();
            for linear in 0..src.numel() {
                if is_nz(linear) {
                    let mut idx = linear;
                    let mut coords = vec![0i64; ndim];
                    for axis in (0..ndim).rev() {
                        coords[axis] = (idx % shape[axis]) as i64;
                        idx /= shape[axis];
                    }
                    acc.extend_from_slice(&coords);
                }
            }
            acc
        };
        match src.dtype() {
            Dtype::F32 => {
                let it = src
                    .iter_elements::<f32>()
                    .expect("dtype")
                    .collect::<Vec<f32>>();
                nonzero_coords = walk(&|i| it[i] != 0.0 && !it[i].is_nan());
            },
            Dtype::F64 => {
                let it = src
                    .iter_elements::<f64>()
                    .expect("dtype")
                    .collect::<Vec<f64>>();
                nonzero_coords = walk(&|i| it[i] != 0.0 && !it[i].is_nan());
            },
            Dtype::I64 => {
                let it = src
                    .iter_elements::<i64>()
                    .expect("dtype")
                    .collect::<Vec<i64>>();
                nonzero_coords = walk(&|i| it[i] != 0);
            },
            Dtype::I32 => {
                let it = src
                    .iter_elements::<i32>()
                    .expect("dtype")
                    .collect::<Vec<i32>>();
                nonzero_coords = walk(&|i| it[i] != 0);
            },
            Dtype::Bool => {
                let it = src
                    .iter_elements::<bool>()
                    .expect("dtype")
                    .collect::<Vec<bool>>();
                nonzero_coords = walk(&|i| it[i]);
            },
            d => {
                return Err(BackendError::DtypeMismatch {
                    op: "nonzero",
                    lhs: d,
                    rhs: d,
                })
            },
        };
        let n = nonzero_coords.len() / ndim.max(1);
        let final_shape = if ndim == 0 { vec![n] } else { vec![n, ndim] };
        Tensor::from_vec_typed::<i64, _>(final_shape, nonzero_coords).map_err(|_| {
            BackendError::OutOfMemory {
                bytes: n * ndim * core::mem::size_of::<i64>(),
            }
        })
    }

    // -------------------- Reductions (P1.4) --------------------

    fn sum_dim(&self, src: &Tensor, dims: &[usize], keepdim: bool) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::sum_dim(src, dims, keepdim)
    }
    fn mean_dim(
        &self,
        src: &Tensor,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::mean_dim(src, dims, keepdim)
    }
    fn max_dim(&self, src: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::max_dim(src, dim, keepdim)
    }
    fn min_dim(&self, src: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::min_dim(src, dim, keepdim)
    }
    fn argmax(&self, src: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::argmax(src, dim, keepdim)
    }
    fn argmin(&self, src: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::argmin(src, dim, keepdim)
    }
    fn var_dim(
        &self,
        src: &Tensor,
        dim: usize,
        unbiased: bool,
        keepdim: bool,
    ) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::var_dim(src, dim, unbiased, keepdim)
    }
    fn prod(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::prod(src)
    }
    fn max(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::max(src)
    }
    fn min(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::min(src)
    }
    fn all(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::all(src)
    }
    fn any(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::any(src)
    }
    fn cumsum(&self, src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::cumsum(src, dim)
    }
    fn cumprod(&self, src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
        crate::kernels::reduction::cumprod(src, dim)
    }

    // -------------------- Softmax (P1.4) --------------------

    fn softmax(&self, src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
        crate::kernels::softmax::softmax(src, dim)
    }
    fn log_softmax(&self, src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
        crate::kernels::softmax::log_softmax(src, dim)
    }

    // -------------------- Loss functions (P1.4) --------------------

    fn mse_loss(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: crate::backend::Reduction,
    ) -> Result<Tensor, BackendError> {
        crate::kernels::loss::mse_loss(input, target, reduction)
    }
    fn cross_entropy(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: crate::backend::Reduction,
    ) -> Result<Tensor, BackendError> {
        crate::kernels::loss::cross_entropy(input, target, reduction)
    }
    fn nll_loss(
        &self,
        log_probs: &Tensor,
        target: &Tensor,
        reduction: crate::backend::Reduction,
    ) -> Result<Tensor, BackendError> {
        crate::kernels::loss::nll_loss(log_probs, target, reduction)
    }
    fn bce_with_logits(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: crate::backend::Reduction,
    ) -> Result<Tensor, BackendError> {
        crate::kernels::loss::bce_with_logits(input, target, reduction)
    }

    // -------------------- Shape ops (delegate to kernels::shape_ops) --------------------

    fn cat(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor, BackendError> {
        crate::kernels::shape_ops::cat(tensors, dim)
    }

    fn stack(&self, tensors: &[&Tensor], dim: usize) -> Result<Tensor, BackendError> {
        crate::kernels::shape_ops::stack(tensors, dim)
    }

    fn split(
        &self,
        src: &Tensor,
        split_size: usize,
        dim: usize,
    ) -> Result<Vec<Tensor>, BackendError> {
        crate::kernels::shape_ops::split(src, split_size, dim)
    }

    fn chunk(
        &self,
        src: &Tensor,
        n_chunks: usize,
        dim: usize,
    ) -> Result<Vec<Tensor>, BackendError> {
        crate::kernels::shape_ops::chunk(src, n_chunks, dim)
    }

    fn repeat(&self, src: &Tensor, repeats: &[usize]) -> Result<Tensor, BackendError> {
        crate::kernels::shape_ops::repeat(src, repeats)
    }

    fn flip(&self, src: &Tensor, dims: &[usize]) -> Result<Tensor, BackendError> {
        crate::kernels::shape_ops::flip(src, dims)
    }

    fn roll(&self, src: &Tensor, shifts: i64, dim: usize) -> Result<Tensor, BackendError> {
        crate::kernels::shape_ops::roll(src, shifts, dim)
    }

    fn cast(&self, src: &Tensor, target: Dtype) -> Result<Tensor, BackendError> {
        // Tensor::to_dtype already implements the full 8x8 matrix with
        // saturating semantics: Rust 1.45+ defines `f as i` as
        // saturating (NaN → 0, +Inf → MAX, -Inf → MIN), and Bool
        // conversions are explicit (any nonzero → true; true → 1.0,
        // false → 0.0). We just delegate.
        Ok(src.to_dtype(target))
    }

    // -------------------- A3 — autograd dispatch plumbing (P3.Y) --------------------

    fn bmm(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        bmm_impl(lhs, rhs)
    }

    fn transpose(&self, src: &Tensor, d0: usize, d1: usize) -> Result<Tensor, BackendError> {
        let view = src
            .transpose(d0, d1)
            .map_err(|e| BackendError::NumericalError(format!("transpose: {e}")))?;
        Ok(view.contiguous())
    }

    fn reshape(&self, src: &Tensor, shape: &[usize]) -> Result<Tensor, BackendError> {
        let numel: usize = shape.iter().product();
        if numel != src.numel() {
            return Err(BackendError::ShapeMismatch {
                op: "reshape",
                lhs: src.shape().to_vec(),
                rhs: shape.to_vec(),
            });
        }
        let data = src.as_slice::<f32>().ok_or_else(|| {
            BackendError::NumericalError("reshape: expected contiguous F32 source".to_string())
        })?;
        Tensor::from_vec(shape.to_vec(), data.to_vec())
            .map_err(|e| BackendError::NumericalError(format!("reshape build: {e}")))
    }

    fn add_bias(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor, BackendError> {
        if x.ndim() != 2 || bias.ndim() != 1 {
            return Err(BackendError::ShapeMismatch {
                op: "add_bias",
                lhs: x.shape().to_vec(),
                rhs: bias.shape().to_vec(),
            });
        }
        let batch = x.shape()[0];
        let n_out = x.shape()[1];
        if bias.shape() != [n_out] {
            return Err(BackendError::ShapeMismatch {
                op: "add_bias",
                lhs: x.shape().to_vec(),
                rhs: bias.shape().to_vec(),
            });
        }
        let bias_buf: &[f32] = bias.as_slice::<f32>().ok_or_else(|| {
            BackendError::NumericalError("add_bias: expected contiguous F32 bias".to_string())
        })?;
        // Native parallel broadcast: write `out[b, j] = x[b, j] + bias[j]`
        // directly without materialising a `[batch, n_out]` bias_wide
        // tensor (which previously cost an `extend_from_slice` loop +
        // a second `self.add` pass + two intermediate allocs).
        if let Some(x_buf) = x.as_slice::<f32>() {
            use rayon::prelude::*;
            let n = batch * n_out;
            let mut out: Vec<core::mem::MaybeUninit<f32>> = Vec::with_capacity(n);
            // SAFETY: capacity is exactly n; every slot is written below.
            #[allow(clippy::uninit_vec)]
            unsafe {
                out.set_len(n);
            }
            const PARALLEL_THRESHOLD: usize = 16_384;
            if n < PARALLEL_THRESHOLD {
                for b in 0..batch {
                    for j in 0..n_out {
                        out[b * n_out + j].write(x_buf[b * n_out + j] + bias_buf[j]);
                    }
                }
            } else {
                out.par_chunks_mut(n_out)
                    .zip(x_buf.par_chunks(n_out))
                    .for_each(|(out_row, x_row)| {
                        for j in 0..n_out {
                            out_row[j].write(x_row[j] + bias_buf[j]);
                        }
                    });
            }
            // SAFETY: every slot written above.
            let out: Vec<f32> = unsafe {
                let mut o = core::mem::ManuallyDrop::new(out);
                Vec::from_raw_parts(o.as_mut_ptr() as *mut f32, n, o.capacity())
            };
            return Tensor::from_vec_typed::<f32, _>(vec![batch, n_out], out).map_err(|_| {
                BackendError::OutOfMemory {
                    bytes: n * core::mem::size_of::<f32>(),
                }
            });
        }
        // Fallback for non-contiguous x.
        let mut wide = Vec::with_capacity(batch * n_out);
        for _ in 0..batch {
            wide.extend_from_slice(bias_buf);
        }
        let bias_wide = Tensor::from_vec([batch, n_out], wide)
            .map_err(|e| BackendError::NumericalError(format!("add_bias broadcast: {e}")))?;
        self.add(x, &bias_wide)
    }

    fn unbroadcast_to(
        &self,
        grad: &Tensor,
        target_shape: &[usize],
    ) -> Result<Tensor, BackendError> {
        unbroadcast_to_impl(self, grad, target_shape)
    }

    fn softmax_grad(
        &self,
        grad: &Tensor,
        output: &Tensor,
        dim: usize,
    ) -> Result<Tensor, BackendError> {
        // d_input = output * (grad - sum(grad * output, dim, keepdim=true))
        let prod = self.mul(grad, output)?;
        let sum_keep = self.sum_dim(&prod, &[dim], true)?;
        let diff = self.sub(grad, &sum_keep)?;
        self.mul(output, &diff)
    }
}

/// Compare-kinds shared by eq/ne/lt/le/gt/ge.
#[derive(Debug, Clone, Copy)]
enum CmpKind {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

fn cmp(
    lhs: &Tensor,
    rhs: &Tensor,
    op_name: &'static str,
    kind: CmpKind,
) -> Result<Tensor, BackendError> {
    if lhs.dtype() != rhs.dtype() {
        return Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        });
    }
    macro_rules! dispatch {
        ($t:ty) => {{
            match kind {
                CmpKind::Eq => map_binary::<$t, bool, _>(lhs, rhs, op_name, |a, b| a == b),
                CmpKind::Ne => map_binary::<$t, bool, _>(lhs, rhs, op_name, |a, b| a != b),
                CmpKind::Lt => map_binary::<$t, bool, _>(lhs, rhs, op_name, |a, b| a < b),
                CmpKind::Le => map_binary::<$t, bool, _>(lhs, rhs, op_name, |a, b| a <= b),
                CmpKind::Gt => map_binary::<$t, bool, _>(lhs, rhs, op_name, |a, b| a > b),
                CmpKind::Ge => map_binary::<$t, bool, _>(lhs, rhs, op_name, |a, b| a >= b),
            }
        }};
    }
    match lhs.dtype() {
        Dtype::F32 => dispatch!(f32),
        Dtype::F64 => dispatch!(f64),
        Dtype::I64 => dispatch!(i64),
        Dtype::I32 => dispatch!(i32),
        Dtype::I8 => dispatch!(i8),
        Dtype::Bool => dispatch!(bool),
        d => Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: d,
            rhs: d,
        }),
    }
}

// -------------------- Indexing kernel helpers --------------------

fn gather_typed<T: rustorch_core::tensor::dtype::Element>(
    src: &Tensor,
    dim: usize,
    idx: &Tensor,
) -> Result<Tensor, BackendError> {
    let out_shape = idx.shape().to_vec();
    let n_out = out_shape.iter().product::<usize>();
    let src_dim_size = src.shape()[dim];
    let src_typed = src.iter_elements::<T>().expect("dtype").collect::<Vec<T>>();
    let idx_typed = idx
        .iter_elements::<i64>()
        .expect("idx i64")
        .collect::<Vec<i64>>();

    let mut out: Vec<T> = Vec::with_capacity(n_out);
    let src_strides = contiguous_strides(src.shape());
    for (linear, &i_dim) in idx_typed.iter().enumerate().take(n_out) {
        let coords = decode_coords(linear, &out_shape);
        if i_dim < 0 || (i_dim as usize) >= src_dim_size {
            return Err(BackendError::IndexOutOfBounds {
                op: "gather",
                index: i_dim,
                bound: src_dim_size,
            });
        }
        let mut src_coords = coords.clone();
        src_coords[dim] = i_dim as usize;
        let src_linear: usize = src_coords
            .iter()
            .zip(src_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        out.push(src_typed[src_linear]);
    }
    Tensor::from_vec_typed::<T, _>(out_shape, out).map_err(|_| BackendError::OutOfMemory {
        bytes: n_out * core::mem::size_of::<T>(),
    })
}

fn scatter_dispatch(
    dst: &Tensor,
    dim: usize,
    idx: &Tensor,
    src: &Tensor,
    op_name: &'static str,
    accumulate: bool,
) -> Result<Tensor, BackendError> {
    if idx.dtype() != Dtype::I64 {
        return Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: idx.dtype(),
            rhs: Dtype::I64,
        });
    }
    if dst.dtype() != src.dtype() {
        return Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: dst.dtype(),
            rhs: src.dtype(),
        });
    }
    if dst.ndim() != idx.ndim() || dst.ndim() != src.ndim() {
        return Err(BackendError::ShapeMismatch {
            op: op_name,
            lhs: dst.shape().to_vec(),
            rhs: idx.shape().to_vec(),
        });
    }
    if dim >= dst.ndim() {
        return Err(BackendError::IndexOutOfBounds {
            op: op_name,
            index: dim as i64,
            bound: dst.ndim(),
        });
    }
    if idx.shape() != src.shape() {
        return Err(BackendError::ShapeMismatch {
            op: op_name,
            lhs: idx.shape().to_vec(),
            rhs: src.shape().to_vec(),
        });
    }
    match dst.dtype() {
        Dtype::F32 => scatter_typed::<f32>(dst, dim, idx, src, op_name, accumulate, |a, b| a + b),
        Dtype::F64 => scatter_typed::<f64>(dst, dim, idx, src, op_name, accumulate, |a, b| a + b),
        Dtype::I64 => scatter_typed::<i64>(dst, dim, idx, src, op_name, accumulate, |a, b| a + b),
        Dtype::I32 => scatter_typed::<i32>(dst, dim, idx, src, op_name, accumulate, |a, b| a + b),
        d => Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: d,
            rhs: d,
        }),
    }
}

fn scatter_typed<T: rustorch_core::tensor::dtype::Element>(
    dst: &Tensor,
    dim: usize,
    idx: &Tensor,
    src: &Tensor,
    op_name: &'static str,
    accumulate: bool,
    add: impl Fn(T, T) -> T,
) -> Result<Tensor, BackendError> {
    let dst_shape = dst.shape().to_vec();
    let mut dst_typed = dst.iter_elements::<T>().expect("dtype").collect::<Vec<T>>();
    let idx_typed = idx
        .iter_elements::<i64>()
        .expect("idx i64")
        .collect::<Vec<i64>>();
    let src_typed = src.iter_elements::<T>().expect("dtype").collect::<Vec<T>>();
    let dst_strides = contiguous_strides(&dst_shape);
    let dst_dim_size = dst_shape[dim];
    for linear in 0..idx.numel() {
        let coords = decode_coords(linear, idx.shape());
        let i_dim = idx_typed[linear];
        if i_dim < 0 || (i_dim as usize) >= dst_dim_size {
            return Err(BackendError::IndexOutOfBounds {
                op: op_name,
                index: i_dim,
                bound: dst_dim_size,
            });
        }
        let mut dst_coords = coords;
        dst_coords[dim] = i_dim as usize;
        let dst_linear: usize = dst_coords
            .iter()
            .zip(dst_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        if accumulate {
            dst_typed[dst_linear] = add(dst_typed[dst_linear], src_typed[linear]);
        } else {
            dst_typed[dst_linear] = src_typed[linear];
        }
    }
    Tensor::from_vec_typed::<T, _>(dst_shape, dst_typed).map_err(|_| BackendError::OutOfMemory {
        bytes: dst.numel() * core::mem::size_of::<T>(),
    })
}

fn index_select_typed<T: rustorch_core::tensor::dtype::Element>(
    src: &Tensor,
    dim: usize,
    indices: &Tensor,
) -> Result<Tensor, BackendError> {
    let src_typed = src.iter_elements::<T>().expect("dtype").collect::<Vec<T>>();
    let idx_vec = indices
        .iter_elements::<i64>()
        .expect("idx i64")
        .collect::<Vec<i64>>();
    let mut out_shape = src.shape().to_vec();
    out_shape[dim] = idx_vec.len();
    let n_out = out_shape.iter().product::<usize>();
    let src_strides = contiguous_strides(src.shape());
    let src_dim_size = src.shape()[dim];
    let mut out: Vec<T> = Vec::with_capacity(n_out);
    for linear in 0..n_out {
        let coords = decode_coords(linear, &out_shape);
        let j = idx_vec[coords[dim]];
        if j < 0 || (j as usize) >= src_dim_size {
            return Err(BackendError::IndexOutOfBounds {
                op: "index_select",
                index: j,
                bound: src_dim_size,
            });
        }
        let mut src_coords = coords;
        src_coords[dim] = j as usize;
        let src_linear: usize = src_coords
            .iter()
            .zip(src_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        out.push(src_typed[src_linear]);
    }
    Tensor::from_vec_typed::<T, _>(out_shape, out).map_err(|_| BackendError::OutOfMemory {
        bytes: n_out * core::mem::size_of::<T>(),
    })
}

fn masked_select_typed<T: rustorch_core::tensor::dtype::Element>(
    src: &Tensor,
    mask: &Tensor,
) -> Result<Tensor, BackendError> {
    let src_v = src.iter_elements::<T>().expect("dtype").collect::<Vec<T>>();
    let mask_v = mask
        .iter_elements::<bool>()
        .expect("bool")
        .collect::<Vec<bool>>();
    let mut out: Vec<T> = Vec::new();
    for (v, m) in src_v.iter().zip(mask_v.iter()) {
        if *m {
            out.push(*v);
        }
    }
    Tensor::from_vec_typed::<T, _>([out.len()], out)
        .map_err(|_| BackendError::OutOfMemory { bytes: 0 })
}

fn masked_fill_typed<T: rustorch_core::tensor::dtype::Element>(
    src: &Tensor,
    mask: &Tensor,
    value: T,
) -> Result<Tensor, BackendError> {
    let src_v = src.iter_elements::<T>().expect("dtype").collect::<Vec<T>>();
    let mask_v = mask
        .iter_elements::<bool>()
        .expect("bool")
        .collect::<Vec<bool>>();
    let out: Vec<T> = src_v
        .into_iter()
        .zip(mask_v)
        .map(|(v, m)| if m { value } else { v })
        .collect();
    Tensor::from_vec_typed::<T, _>(src.shape().to_vec(), out).map_err(|_| {
        BackendError::OutOfMemory {
            bytes: src.numel() * core::mem::size_of::<T>(),
        }
    })
}

fn where_typed<T: rustorch_core::tensor::dtype::Element>(
    cond: &Tensor,
    x: &Tensor,
    y: &Tensor,
) -> Result<Tensor, BackendError> {
    let cond_v = cond
        .iter_elements::<bool>()
        .expect("bool")
        .collect::<Vec<bool>>();
    let x_v = x.iter_elements::<T>().expect("dtype").collect::<Vec<T>>();
    let y_v = y.iter_elements::<T>().expect("dtype").collect::<Vec<T>>();
    let out: Vec<T> = cond_v
        .iter()
        .zip(x_v.iter().zip(y_v.iter()))
        .map(|(&c, (&xi, &yi))| if c { xi } else { yi })
        .collect();
    Tensor::from_vec_typed::<T, _>(x.shape().to_vec(), out).map_err(|_| BackendError::OutOfMemory {
        bytes: x.numel() * core::mem::size_of::<T>(),
    })
}

/// Compute row-major contiguous element-strides for a shape.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Decode a linear index into multi-dim coords (row-major).
fn decode_coords(linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut idx = linear;
    let mut coords = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = idx % shape[axis];
        idx /= shape[axis];
    }
    coords
}

/// Abramowitz-Stegun 7.1.26 approximation of the error function.
/// Max absolute error ≤ 1.5e-7 over the whole real line.
fn erf_f32(x: f32) -> f32 {
    // erf is odd: erf(-x) = -erf(x).
    let sign = if x < 0.0 { -1.0_f32 } else { 1.0 };
    let a = x.abs();
    // A&S coefficients
    const P: f32 = 0.327_591_1;
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_7;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    let t = 1.0 / (1.0 + P * a);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-a * a).exp();
    sign * y
}

/// Same A&S 7.1.26 in f64 — slightly higher precision (≤ 1.5e-7 still).
fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0_f64 } else { 1.0 };
    let a = x.abs();
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    let t = 1.0 / (1.0 + P * a);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-a * a).exp();
    sign * y
}

/// Helper: dispatch a unary float-only op (f32 + f64 paths).
fn unary_float(
    src: &Tensor,
    op_name: &'static str,
    f32_fn: impl Fn(f32) -> f32 + Sync + Send,
    f64_fn: impl Fn(f64) -> f64 + Sync + Send,
) -> Result<Tensor, BackendError> {
    match src.dtype() {
        Dtype::F32 => map_unary_same::<f32, _>(src, op_name, f32_fn),
        Dtype::F64 => map_unary_same::<f64, _>(src, op_name, f64_fn),
        d => Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: d,
            rhs: d,
        }),
    }
}

/// Discriminator for the dispatch_binary helper.
enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

fn dispatch_binary(
    lhs: &Tensor,
    rhs: &Tensor,
    op_name: &'static str,
    kind: BinaryKind,
) -> Result<Tensor, BackendError> {
    if lhs.dtype() != rhs.dtype() {
        return Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        });
    }
    match (lhs.dtype(), &kind) {
        (Dtype::F32, BinaryKind::Add) => map_binary_same::<f32, _>(lhs, rhs, op_name, |a, b| a + b),
        (Dtype::F32, BinaryKind::Sub) => map_binary_same::<f32, _>(lhs, rhs, op_name, |a, b| a - b),
        (Dtype::F32, BinaryKind::Mul) => map_binary_same::<f32, _>(lhs, rhs, op_name, |a, b| a * b),
        (Dtype::F32, BinaryKind::Div) => map_binary_same::<f32, _>(lhs, rhs, op_name, |a, b| a / b),
        (Dtype::F64, BinaryKind::Add) => map_binary_same::<f64, _>(lhs, rhs, op_name, |a, b| a + b),
        (Dtype::F64, BinaryKind::Sub) => map_binary_same::<f64, _>(lhs, rhs, op_name, |a, b| a - b),
        (Dtype::F64, BinaryKind::Mul) => map_binary_same::<f64, _>(lhs, rhs, op_name, |a, b| a * b),
        (Dtype::F64, BinaryKind::Div) => map_binary_same::<f64, _>(lhs, rhs, op_name, |a, b| a / b),
        (Dtype::I64, BinaryKind::Add) => map_binary_same::<i64, _>(lhs, rhs, op_name, |a, b| a + b),
        (Dtype::I64, BinaryKind::Sub) => map_binary_same::<i64, _>(lhs, rhs, op_name, |a, b| a - b),
        (Dtype::I64, BinaryKind::Mul) => map_binary_same::<i64, _>(lhs, rhs, op_name, |a, b| a * b),
        (Dtype::I64, BinaryKind::Div) => {
            map_binary_same::<i64, _>(lhs, rhs, op_name, |a, b| a.checked_div(b).unwrap_or(0))
        },
        (Dtype::I32, BinaryKind::Add) => map_binary_same::<i32, _>(lhs, rhs, op_name, |a, b| a + b),
        (Dtype::I32, BinaryKind::Sub) => map_binary_same::<i32, _>(lhs, rhs, op_name, |a, b| a - b),
        (Dtype::I32, BinaryKind::Mul) => map_binary_same::<i32, _>(lhs, rhs, op_name, |a, b| a * b),
        (Dtype::I32, BinaryKind::Div) => {
            map_binary_same::<i32, _>(lhs, rhs, op_name, |a, b| a.checked_div(b).unwrap_or(0))
        },
        // BF16 / F16 — accumulate via f32 since `half::bf16` does not
        // implement `Add` directly. Element-wise: cast → op → cast back.
        (Dtype::BF16, BinaryKind::Add) => {
            map_binary_same::<half::bf16, _>(lhs, rhs, op_name, |a, b| {
                half::bf16::from_f32(a.to_f32() + b.to_f32())
            })
        },
        (Dtype::BF16, BinaryKind::Sub) => {
            map_binary_same::<half::bf16, _>(lhs, rhs, op_name, |a, b| {
                half::bf16::from_f32(a.to_f32() - b.to_f32())
            })
        },
        (Dtype::BF16, BinaryKind::Mul) => {
            map_binary_same::<half::bf16, _>(lhs, rhs, op_name, |a, b| {
                half::bf16::from_f32(a.to_f32() * b.to_f32())
            })
        },
        (Dtype::BF16, BinaryKind::Div) => {
            map_binary_same::<half::bf16, _>(lhs, rhs, op_name, |a, b| {
                half::bf16::from_f32(a.to_f32() / b.to_f32())
            })
        },
        (Dtype::F16, BinaryKind::Add) => {
            map_binary_same::<half::f16, _>(lhs, rhs, op_name, |a, b| {
                half::f16::from_f32(a.to_f32() + b.to_f32())
            })
        },
        (Dtype::F16, BinaryKind::Sub) => {
            map_binary_same::<half::f16, _>(lhs, rhs, op_name, |a, b| {
                half::f16::from_f32(a.to_f32() - b.to_f32())
            })
        },
        (Dtype::F16, BinaryKind::Mul) => {
            map_binary_same::<half::f16, _>(lhs, rhs, op_name, |a, b| {
                half::f16::from_f32(a.to_f32() * b.to_f32())
            })
        },
        (Dtype::F16, BinaryKind::Div) => {
            map_binary_same::<half::f16, _>(lhs, rhs, op_name, |a, b| {
                half::f16::from_f32(a.to_f32() / b.to_f32())
            })
        },
        (d, _) => Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: d,
            rhs: d,
        }),
    }
}

/// Threshold above which we dispatch to the SIMD-vectorized `gemm` crate.
/// Below this, the dispatch overhead dominates the work — a register-tile
/// scalar loop is faster. Calibrated empirically on Apple M4 Max P-core.
const GEMM_DISPATCH_MIN: usize = 32;

/// f32 matmul dispatch.
///
/// Priority order:
/// 1. **macOS**: Apple `Accelerate.framework` `cblas_sgemm` — routes
///    through AMX tile units automatically on Apple Silicon, typically
///    5-10× faster than pure-Rust `gemm`-rs on the same hardware. This
///    is the canonical CPU matmul path on macOS (PyTorch CPU since 2.3+
///    uses the same backend).
/// 2. **Other platforms**: pure-Rust `gemm` 0.18 (faer-rs ecosystem),
///    NEON+AVX-512 explicit vectorization, ~70-80% Intel MKL on x86.
/// 3. **wasm32**: scalar [`matmul_naive`] fallback (no SIMD intrinsics
///    available).
///
/// Below [`GEMM_DISPATCH_MIN`], the scalar fallback is used regardless
/// of platform: BLAS setup overhead exceeds compute for tiny shapes.
#[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
fn matmul_dispatch_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor, BackendError> {
    if m < GEMM_DISPATCH_MIN || n < GEMM_DISPATCH_MIN || k < GEMM_DISPATCH_MIN {
        return matmul_naive::<f32>(lhs, rhs, m, k, n);
    }
    // Fast path: borrow contiguous tensor data directly. The
    // `as_slice::<f32>()` call returns `Some` only when the tensor is
    // contiguous + f32 + offset 0 — exactly the conditions cblas_sgemm
    // needs. Avoids ~8 MB of `iter_elements().collect()` allocations
    // per matmul (3× per training step) and the corresponding heap
    // pressure.
    let lhs_owned: Option<Vec<f32>>;
    let rhs_owned: Option<Vec<f32>>;
    let lhs_slice: &[f32] = if let Some(s) = lhs.as_slice::<f32>() {
        lhs_owned = None;
        s
    } else {
        // Non-contiguous (transpose view, etc.) — materialise once.
        lhs_owned = Some(
            lhs.iter_elements::<f32>()
                .ok_or(BackendError::DtypeMismatch {
                    op: "matmul",
                    lhs: lhs.dtype(),
                    rhs: rhs.dtype(),
                })?
                .collect(),
        );
        lhs_owned.as_deref().unwrap()
    };
    let rhs_slice: &[f32] = if let Some(s) = rhs.as_slice::<f32>() {
        rhs_owned = None;
        s
    } else {
        rhs_owned = Some(
            rhs.iter_elements::<f32>()
                .ok_or(BackendError::DtypeMismatch {
                    op: "matmul",
                    lhs: lhs.dtype(),
                    rhs: rhs.dtype(),
                })?
                .collect(),
        );
        rhs_owned.as_deref().unwrap()
    };
    let mut out_buf: Vec<f32> = vec![0.0_f32; m * n];

    // SAFETY: input slices have length ≥ m*k / k*n verified by Tensor's
    // contiguity invariant; out_buf is exactly m*n. `cblas_sgemm` is the
    // canonical row-major f32 GEMM ABI; the wrapper validates lengths
    // in debug builds.
    unsafe {
        crate::accelerate::sgemm_row_major(m, k, n, lhs_slice, rhs_slice, &mut out_buf);
    }
    drop(lhs_owned);
    drop(rhs_owned);

    Tensor::from_vec_typed::<f32, _>(vec![m, n], out_buf).map_err(|_| BackendError::OutOfMemory {
        bytes: m * n * core::mem::size_of::<f32>(),
    })
}

/// f32 matmul dispatch (non-macOS): pure-Rust `gemm`-rs.
#[cfg(all(not(target_os = "macos"), not(target_arch = "wasm32")))]
fn matmul_dispatch_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor, BackendError> {
    if m < GEMM_DISPATCH_MIN || n < GEMM_DISPATCH_MIN || k < GEMM_DISPATCH_MIN {
        return matmul_naive::<f32>(lhs, rhs, m, k, n);
    }
    let lhs_buf: Vec<f32> = lhs
        .iter_elements::<f32>()
        .ok_or(BackendError::DtypeMismatch {
            op: "matmul",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        })?
        .collect();
    let rhs_buf: Vec<f32> = rhs
        .iter_elements::<f32>()
        .ok_or(BackendError::DtypeMismatch {
            op: "matmul",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        })?
        .collect();
    let mut out_buf: Vec<f32> = vec![0.0_f32; m * n];

    // Row-major [m, k] @ [k, n] -> [m, n]:
    //   strides for the gemm crate (in elements, not bytes):
    //   - lhs: rs=k, cs=1
    //   - rhs: rs=n, cs=1
    //   - dst: rs=n, cs=1
    // SAFETY: buffers are exactly m*k, k*n, m*n long in f32 and live for
    // the duration of the call. The gemm crate is `unsafe fn` because it
    // works through raw pointers, not because of additional invariants.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out_buf.as_mut_ptr(),
            1,
            n as isize,
            false,
            lhs_buf.as_ptr(),
            1,
            k as isize,
            rhs_buf.as_ptr(),
            1,
            n as isize,
            0.0_f32,
            1.0_f32,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }

    Tensor::from_vec_typed::<f32, _>(vec![m, n], out_buf).map_err(|_| BackendError::OutOfMemory {
        bytes: m * n * core::mem::size_of::<f32>(),
    })
}

/// wasm32 fallback for `matmul_dispatch_f32` — `gemm` 0.18 doesn't
/// support wasm32 (NEON/AVX intrinsics) so we always go through the
/// scalar [`matmul_naive`] kernel.
#[cfg(target_arch = "wasm32")]
fn matmul_dispatch_f32(
    lhs: &Tensor,
    rhs: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor, BackendError> {
    matmul_naive::<f32>(lhs, rhs, m, k, n)
}

/// f64 matmul dispatch: SIMD `gemm` crate above [`GEMM_DISPATCH_MIN`],
/// scalar fallback otherwise.
#[cfg(not(target_arch = "wasm32"))]
fn matmul_dispatch_f64(
    lhs: &Tensor,
    rhs: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor, BackendError> {
    if m < GEMM_DISPATCH_MIN || n < GEMM_DISPATCH_MIN || k < GEMM_DISPATCH_MIN {
        return matmul_naive::<f64>(lhs, rhs, m, k, n);
    }
    let lhs_buf: Vec<f64> = lhs
        .iter_elements::<f64>()
        .ok_or(BackendError::DtypeMismatch {
            op: "matmul",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        })?
        .collect();
    let rhs_buf: Vec<f64> = rhs
        .iter_elements::<f64>()
        .ok_or(BackendError::DtypeMismatch {
            op: "matmul",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        })?
        .collect();
    let mut out_buf: Vec<f64> = vec![0.0_f64; m * n];
    // SAFETY: see matmul_dispatch_f32 — same invariants.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out_buf.as_mut_ptr(),
            1,
            n as isize,
            false,
            lhs_buf.as_ptr(),
            1,
            k as isize,
            rhs_buf.as_ptr(),
            1,
            n as isize,
            0.0_f64,
            1.0_f64,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }
    Tensor::from_vec_typed::<f64, _>(vec![m, n], out_buf).map_err(|_| BackendError::OutOfMemory {
        bytes: m * n * core::mem::size_of::<f64>(),
    })
}

/// wasm32 fallback for `matmul_dispatch_f64` — scalar [`matmul_naive`].
#[cfg(target_arch = "wasm32")]
fn matmul_dispatch_f64(
    lhs: &Tensor,
    rhs: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor, BackendError> {
    matmul_naive::<f64>(lhs, rhs, m, k, n)
}

/// Generic naïve `O(M*K*N)` matmul. Walks contiguous-or-not via
/// strided index offset (`m` = lhs shape[0], `k` = lhs shape[1], `n` =
/// rhs shape[1]).
fn matmul_naive<T>(
    lhs: &Tensor,
    rhs: &Tensor,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor, BackendError>
where
    T: rustorch_core::tensor::dtype::Element
        + Default
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>,
{
    let mut out: Vec<T> = vec![T::default(); m * n];
    let lhs_strides = lhs.strides().to_vec();
    let rhs_strides = rhs.strides().to_vec();
    let lhs_off = lhs.storage_offset() as isize;
    let rhs_off = rhs.storage_offset() as isize;
    // SAFETY: dtype matches by call-site dispatch.
    let lhs_raw: &[T] = unsafe { lhs.storage().as_slice::<T>() };
    let rhs_raw: &[T] = unsafe { rhs.storage().as_slice::<T>() };
    for i in 0..m {
        for j in 0..n {
            let mut acc: T = T::default();
            for kk in 0..k {
                let li =
                    (lhs_off + i as isize * lhs_strides[0] + kk as isize * lhs_strides[1]) as usize;
                let ri =
                    (rhs_off + kk as isize * rhs_strides[0] + j as isize * rhs_strides[1]) as usize;
                acc = acc + lhs_raw[li] * rhs_raw[ri];
            }
            out[i * n + j] = acc;
        }
    }
    Tensor::from_vec_typed::<T, _>(vec![m, n], out).map_err(|_| BackendError::OutOfMemory {
        bytes: m * n * core::mem::size_of::<T>(),
    })
}

/// Generic reduction kernel: walks the source in shape order via
/// [`Tensor::iter_elements`], folding with `f`.
fn sum_kernel<T>(src: &Tensor, init: T, f: impl Fn(T, T) -> T) -> Result<Tensor, BackendError>
where
    T: rustorch_core::tensor::dtype::Element,
{
    let it = src
        .iter_elements::<T>()
        .ok_or(BackendError::DtypeMismatch {
            op: "sum",
            lhs: src.dtype(),
            rhs: T::DTYPE,
        })?;
    let mut acc = init;
    for v in it {
        acc = f(acc, v);
    }
    Tensor::from_vec_typed::<T, _>([], vec![acc]).map_err(|_| BackendError::OutOfMemory {
        bytes: core::mem::size_of::<T>(),
    })
}

// -------------------- A3 — autograd dispatch plumbing helpers --------------------

/// CPU bmm: `[B, M, K] @ [B, K, N] = [B, M, N]`. Triple-loop per batch.
/// Ported from `rustorch-autograd/src/ops.rs::bmm_forward` so the kernel
/// becomes part of the Backend trait surface (enables device dispatch).
fn bmm_impl(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
    let l_shape = lhs.shape();
    let r_shape = rhs.shape();
    if l_shape.len() != 3 || r_shape.len() != 3 {
        return Err(BackendError::ShapeMismatch {
            op: "bmm",
            lhs: l_shape.to_vec(),
            rhs: r_shape.to_vec(),
        });
    }
    let (b, m, k1) = (l_shape[0], l_shape[1], l_shape[2]);
    let (b2, k2, n) = (r_shape[0], r_shape[1], r_shape[2]);
    if b != b2 || k1 != k2 {
        return Err(BackendError::ShapeMismatch {
            op: "bmm",
            lhs: l_shape.to_vec(),
            rhs: r_shape.to_vec(),
        });
    }
    let l_data = lhs
        .as_slice::<f32>()
        .ok_or_else(|| BackendError::NumericalError("bmm: lhs must be contiguous F32".into()))?;
    let r_data = rhs
        .as_slice::<f32>()
        .ok_or_else(|| BackendError::NumericalError("bmm: rhs must be contiguous F32".into()))?;
    let k = k1;
    let mut out = vec![0.0_f32; b * m * n];
    for bi in 0..b {
        let l_off = bi * m * k;
        let r_off = bi * k * n;
        let o_off = bi * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    acc += l_data[l_off + i * k + kk] * r_data[r_off + kk * n + j];
                }
                out[o_off + i * n + j] = acc;
            }
        }
    }
    Tensor::from_vec([b, m, n], out)
        .map_err(|e| BackendError::NumericalError(format!("bmm output build: {e}")))
}

/// Sum `grad` across axes that were broadcasted up to its current shape,
/// returning a tensor of `target_shape`.
///
/// Algorithm:
/// 1. Pad `target_shape` on the left with 1s to match `grad.ndim()`.
/// 2. For each axis where the padded target is 1 but `grad.shape[axis]` > 1,
///    sum along that axis with `keepdim=true`.
/// 3. Reshape back to `target_shape` (drops the padded leading dims).
///
/// This is the inverse of NumPy/PyTorch broadcasting and is required by
/// every binary op backward (add/sub/mul/div) when the inputs were
/// broadcasted.
fn unbroadcast_to_impl(
    backend: &CpuBackend,
    grad: &Tensor,
    target_shape: &[usize],
) -> Result<Tensor, BackendError> {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return Ok(grad.clone());
    }
    let g_ndim = grad_shape.len();
    let t_ndim = target_shape.len();
    if t_ndim > g_ndim {
        return Err(BackendError::ShapeMismatch {
            op: "unbroadcast_to",
            lhs: grad_shape.to_vec(),
            rhs: target_shape.to_vec(),
        });
    }
    // Pad target_shape on the left with 1s up to g_ndim.
    let pad = g_ndim - t_ndim;
    let mut padded = vec![1usize; pad];
    padded.extend_from_slice(target_shape);
    // Validate compatibility along each axis: target_padded[i] must be 1
    // or equal to grad_shape[i].
    for (i, (&g, &t)) in grad_shape.iter().zip(padded.iter()).enumerate() {
        if t != 1 && t != g {
            return Err(BackendError::ShapeMismatch {
                op: "unbroadcast_to",
                lhs: grad_shape.to_vec(),
                rhs: target_shape.to_vec(),
            });
        }
        let _ = i;
    }
    // Collect axes to reduce (where padded == 1 and grad > 1).
    let reduce_axes: Vec<usize> = padded
        .iter()
        .zip(grad_shape.iter())
        .enumerate()
        .filter_map(|(axis, (&p, &g))| if p == 1 && g > 1 { Some(axis) } else { None })
        .collect();
    let reduced = if reduce_axes.is_empty() {
        grad.clone()
    } else {
        backend.sum_dim(grad, &reduce_axes, true)?
    };
    // reduced now has shape `padded` (with reduced axes = 1). Reshape to
    // target_shape (drops the leading padded 1s).
    if reduced.shape() == target_shape {
        Ok(reduced)
    } else {
        backend.reshape(&reduced, target_shape)
    }
}

/// Public function returning the static CPU backend singleton.
///
/// Cheap to call repeatedly — `CpuBackend` is a unit type stored in
/// `static`.
pub fn cpu_backend() -> &'static dyn Backend {
    static CPU: CpuBackend = CpuBackend::new();
    &CPU
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b() -> &'static dyn Backend {
        cpu_backend()
    }

    #[test]
    fn name() {
        assert_eq!(b().name(), "cpu");
    }

    #[test]
    fn add_simple() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
        let c = b().add(&a, &a).unwrap();
        assert_eq!(c.as_slice::<f32>().unwrap(), &[2.0_f32, 4.0, 6.0]);
    }

    #[test]
    fn add_broadcast_row() {
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let r = Tensor::from_vec([3usize], vec![10.0_f32, 20.0, 30.0]).unwrap();
        let c = b().add(&a, &r).unwrap();
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[11.0_f32, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    #[test]
    fn sub_mul_div_paths() {
        let a = Tensor::from_vec([3usize], vec![6.0_f32, 8.0, 10.0]).unwrap();
        let z = Tensor::from_vec([3usize], vec![2.0_f32, 4.0, 5.0]).unwrap();
        assert_eq!(
            b().sub(&a, &z).unwrap().as_slice::<f32>().unwrap(),
            &[4.0, 4.0, 5.0]
        );
        assert_eq!(
            b().mul(&a, &z).unwrap().as_slice::<f32>().unwrap(),
            &[12.0, 32.0, 50.0]
        );
        assert_eq!(
            b().div(&a, &z).unwrap().as_slice::<f32>().unwrap(),
            &[3.0, 2.0, 2.0]
        );
    }

    #[test]
    fn neg_paths() {
        let f = Tensor::from_vec([3usize], vec![1.0_f32, -2.0, 3.0]).unwrap();
        assert_eq!(
            b().neg(&f).unwrap().as_slice::<f32>().unwrap(),
            &[-1.0, 2.0, -3.0]
        );
        let i = Tensor::from_vec_typed::<i64, _>([3usize], vec![1_i64, -2, 3]).unwrap();
        assert_eq!(
            b().neg(&i).unwrap().as_slice::<i64>().unwrap(),
            &[-1_i64, 2, -3]
        );
    }

    #[test]
    fn matmul_2x3_3x2_f32() {
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b_t = Tensor::from_vec([3usize, 2], vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = b().matmul(&a, &b_t).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.as_slice::<f32>().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_dim_mismatch_returns_err() {
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
        let b_t = Tensor::from_vec([4usize, 2], vec![1.0_f32; 8]).unwrap();
        assert!(matches!(
            b().matmul(&a, &b_t),
            Err(BackendError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn matmul_rank_not_2_returns_err() {
        let a = Tensor::from_vec([6usize], vec![1.0_f32; 6]).unwrap();
        let b_t = Tensor::from_vec([6usize], vec![1.0_f32; 6]).unwrap();
        assert!(matches!(
            b().matmul(&a, &b_t),
            Err(BackendError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn matmul_against_transposed_view() {
        // Test that matmul works on non-contiguous strides (transpose).
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let a_t = a.transpose(0, 1).unwrap(); // [3, 2], non-contig
                                              // a_t @ a = [3, 2] @ [2, 3] = [3, 3]
        let c = b().matmul(&a_t, &a).unwrap();
        assert_eq!(c.shape(), &[3, 3]);
        // Reference: a_t[i,k] * a[k,j] for i, j in [0, 3)
        //  a_t = [[1,4],[2,5],[3,6]]; a = [[1,2,3],[4,5,6]]
        //  c[0,0] = 1*1+4*4=17; c[0,1] = 1*2+4*5=22; c[0,2] = 1*3+4*6=27
        //  c[1,0] = 2*1+5*4=22; c[1,1] = 2*2+5*5=29; c[1,2] = 2*3+5*6=36
        //  c[2,0] = 3*1+6*4=27; c[2,1] = 3*2+6*5=36; c[2,2] = 3*3+6*6=45
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]
        );
    }

    #[test]
    fn sum_full() {
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(b().sum(&a).unwrap().as_slice::<f32>().unwrap(), &[21.0]);
    }

    #[test]
    fn mean_full() {
        let a = Tensor::from_vec([4usize], vec![2.0_f32, 4.0, 6.0, 8.0]).unwrap();
        assert_eq!(b().mean(&a).unwrap().as_slice::<f32>().unwrap(), &[5.0]);
    }

    #[test]
    fn mean_of_empty_tensor_err() {
        let a = Tensor::zeros([0usize]);
        assert!(matches!(b().mean(&a), Err(BackendError::NumericalError(_))));
    }

    #[test]
    fn relu_clamps_negatives() {
        let a = Tensor::from_vec([4usize], vec![-1.0_f32, 0.0, 1.0, 2.0]).unwrap();
        assert_eq!(
            b().relu(&a).unwrap().as_slice::<f32>().unwrap(),
            &[0.0, 0.0, 1.0, 2.0]
        );
    }

    #[test]
    fn eq_returns_bool() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
        let bb = Tensor::from_vec([3usize], vec![1.0_f32, 2.5, 3.0]).unwrap();
        let c = b().eq(&a, &bb).unwrap();
        assert_eq!(c.dtype(), rustorch_core::tensor::Dtype::Bool);
        assert_eq!(c.as_slice::<bool>().unwrap(), &[true, false, true]);
    }

    #[test]
    fn dtype_mismatch_propagates() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
        let i = Tensor::from_vec_typed::<i64, _>([3usize], vec![1_i64, 2, 3]).unwrap();
        assert!(matches!(
            b().add(&a, &i),
            Err(BackendError::DtypeMismatch { .. })
        ));
    }

    #[test]
    fn shape_mismatch_propagates() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
        let c = Tensor::from_vec([4usize], vec![1.0_f32; 4]).unwrap();
        assert!(matches!(
            b().add(&a, &c),
            Err(BackendError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn singleton_is_consistent() {
        let a = b().name();
        let bn = cpu_backend().name();
        assert_eq!(a, bn);
    }

    // -------------------- P1.3 — extra elementary ops --------------------

    #[test]
    fn abs_f32_and_i64() {
        let a = Tensor::from_vec([3usize], vec![-1.0_f32, 0.0, 2.0]).unwrap();
        assert_eq!(
            b().abs(&a).unwrap().as_slice::<f32>().unwrap(),
            &[1.0, 0.0, 2.0]
        );
        let i = Tensor::from_vec_typed::<i64, _>([3usize], vec![-1_i64, 0, 2]).unwrap();
        assert_eq!(
            b().abs(&i).unwrap().as_slice::<i64>().unwrap(),
            &[1_i64, 0, 2]
        );
    }

    #[test]
    fn sqrt_exp_log() {
        let a = Tensor::from_vec([4usize], vec![1.0_f32, 4.0, 9.0, 16.0]).unwrap();
        let r = b().sqrt(&a).unwrap();
        assert_eq!(r.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        // exp(0) = 1, exp(1) ≈ 2.71828
        let z = Tensor::from_vec([2usize], vec![0.0_f32, 1.0]).unwrap();
        let e = b().exp(&z).unwrap().as_slice::<f32>().unwrap().to_vec();
        assert!((e[0] - 1.0).abs() < 1e-6);
        assert!((e[1] - core::f32::consts::E).abs() < 1e-5);
        // log(e) = 1
        let one = Tensor::from_vec([1usize], vec![core::f32::consts::E]).unwrap();
        let l = b().log(&one).unwrap().as_slice::<f32>().unwrap().to_vec();
        assert!((l[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn trig_at_zero() {
        let z = Tensor::from_vec([1usize], vec![0.0_f32]).unwrap();
        assert_eq!(b().sin(&z).unwrap().as_slice::<f32>().unwrap()[0], 0.0);
        assert_eq!(b().cos(&z).unwrap().as_slice::<f32>().unwrap()[0], 1.0);
        assert_eq!(b().tan(&z).unwrap().as_slice::<f32>().unwrap()[0], 0.0);
    }

    #[test]
    fn pow_scalar_works() {
        let a = Tensor::from_vec([3usize], vec![2.0_f32, 3.0, 4.0]).unwrap();
        let r = b().pow_scalar(&a, 2.0).unwrap();
        assert_eq!(r.as_slice::<f32>().unwrap(), &[4.0, 9.0, 16.0]);
    }

    #[test]
    fn sigmoid_at_zero_is_half() {
        let z = Tensor::from_vec([1usize], vec![0.0_f32]).unwrap();
        let r = b().sigmoid(&z).unwrap().as_slice::<f32>().unwrap()[0];
        assert!((r - 0.5).abs() < 1e-7);
    }

    #[test]
    fn tanh_paths() {
        let z = Tensor::from_vec([3usize], vec![0.0_f32, 1.0, -1.0]).unwrap();
        let r = b().tanh(&z).unwrap().as_slice::<f32>().unwrap().to_vec();
        assert!(r[0].abs() < 1e-7);
        // tanh is odd
        assert!((r[1] + r[2]).abs() < 1e-6);
    }

    #[test]
    fn gelu_at_zero_is_zero() {
        let z = Tensor::from_vec([1usize], vec![0.0_f32]).unwrap();
        let r = b().gelu(&z).unwrap().as_slice::<f32>().unwrap()[0];
        assert!(r.abs() < 1e-6);
    }

    #[test]
    fn leaky_relu_negative_slope() {
        let a = Tensor::from_vec([3usize], vec![-1.0_f32, 0.0, 2.0]).unwrap();
        let r = b().leaky_relu(&a, 0.1).unwrap();
        let v = r.as_slice::<f32>().unwrap();
        assert!((v[0] - (-0.1)).abs() < 1e-6);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 2.0);
    }

    #[test]
    fn silu_at_zero_is_zero() {
        let z = Tensor::from_vec([1usize], vec![0.0_f32]).unwrap();
        let r = b().silu(&z).unwrap().as_slice::<f32>().unwrap()[0];
        assert!(r.abs() < 1e-6);
    }

    #[test]
    fn cmp_lt_le_gt_ge() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
        let bb = Tensor::from_vec([3usize], vec![2.0_f32, 2.0, 2.0]).unwrap();
        assert_eq!(
            b().lt(&a, &bb).unwrap().as_slice::<bool>().unwrap(),
            &[true, false, false]
        );
        assert_eq!(
            b().le(&a, &bb).unwrap().as_slice::<bool>().unwrap(),
            &[true, true, false]
        );
        assert_eq!(
            b().gt(&a, &bb).unwrap().as_slice::<bool>().unwrap(),
            &[false, false, true]
        );
        assert_eq!(
            b().ge(&a, &bb).unwrap().as_slice::<bool>().unwrap(),
            &[false, true, true]
        );
        assert_eq!(
            b().ne(&a, &bb).unwrap().as_slice::<bool>().unwrap(),
            &[true, false, true]
        );
    }

    #[test]
    fn isnan_isinf_isfinite() {
        let t = Tensor::from_vec(
            [4usize],
            vec![1.0_f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        )
        .unwrap();
        assert_eq!(
            b().isnan(&t).unwrap().as_slice::<bool>().unwrap(),
            &[false, true, false, false]
        );
        assert_eq!(
            b().isinf(&t).unwrap().as_slice::<bool>().unwrap(),
            &[false, false, true, true]
        );
        assert_eq!(
            b().isfinite(&t).unwrap().as_slice::<bool>().unwrap(),
            &[true, false, false, false]
        );
    }

    #[test]
    fn unsupported_dtype_returns_err() {
        let t = Tensor::from_vec_typed::<i64, _>([3usize], vec![1_i64, 2, 3]).unwrap();
        assert!(matches!(
            b().sqrt(&t),
            Err(BackendError::DtypeMismatch { .. })
        ));
        assert!(matches!(
            b().sigmoid(&t),
            Err(BackendError::DtypeMismatch { .. })
        ));
    }

    // -------------------- P1.3 Math ops completion --------------------

    #[test]
    fn pow_tensor_tensor() {
        let a = Tensor::from_vec([3usize], vec![2.0_f32, 3.0, 4.0]).unwrap();
        let b_t = Tensor::from_vec([3usize], vec![2.0_f32, 2.0, 2.0]).unwrap();
        let r = b().pow(&a, &b_t).unwrap();
        assert_eq!(r.as_slice::<f32>().unwrap(), &[4.0, 9.0, 16.0]);
    }

    #[test]
    fn rsqrt_basic() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32, 4.0, 16.0]).unwrap();
        let r = b().rsqrt(&a).unwrap();
        let v = r.as_slice::<f32>().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 0.5).abs() < 1e-6);
        assert!((v[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn expm1_log1p_round_trip_near_zero() {
        let a = Tensor::from_vec([3usize], vec![0.0_f32, 1e-7, -1e-7]).unwrap();
        let e = b().expm1(&a).unwrap();
        let v = e.as_slice::<f32>().unwrap();
        assert_eq!(v[0], 0.0);
        assert!((v[1] - 1e-7).abs() < 1e-13);
        let inv = b().log1p(&e).unwrap();
        let w = inv.as_slice::<f32>().unwrap();
        assert!(w[0].abs() < 1e-7);
    }

    #[test]
    fn log2_log10_known_values() {
        let a = Tensor::from_vec([2usize], vec![8.0_f32, 1000.0]).unwrap();
        let r2 = b().log2(&a).unwrap().as_slice::<f32>().unwrap().to_vec();
        let r10 = b().log10(&a).unwrap().as_slice::<f32>().unwrap().to_vec();
        assert!((r2[0] - 3.0).abs() < 1e-5);
        assert!((r10[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn asin_acos_atan_known() {
        let a = Tensor::from_vec([3usize], vec![0.0_f32, 0.5, 1.0]).unwrap();
        let asin_r = b().asin(&a).unwrap().as_slice::<f32>().unwrap().to_vec();
        let acos_r = b().acos(&a).unwrap().as_slice::<f32>().unwrap().to_vec();
        let atan_r = b().atan(&a).unwrap().as_slice::<f32>().unwrap().to_vec();
        assert!(asin_r[0].abs() < 1e-6);
        assert!((asin_r[1] - core::f32::consts::FRAC_PI_6).abs() < 1e-5);
        assert!((asin_r[2] - core::f32::consts::FRAC_PI_2).abs() < 1e-5);
        assert!((acos_r[0] - core::f32::consts::FRAC_PI_2).abs() < 1e-5);
        assert!(acos_r[2].abs() < 1e-6);
        assert!(atan_r[0].abs() < 1e-6);
        assert!((atan_r[2] - core::f32::consts::FRAC_PI_4).abs() < 1e-5);
    }

    #[test]
    fn atan2_y_eq_x_is_pi_over_4() {
        let y = Tensor::from_vec([1usize], vec![1.0_f32]).unwrap();
        let x = Tensor::from_vec([1usize], vec![1.0_f32]).unwrap();
        let r = b().atan2(&y, &x).unwrap().as_slice::<f32>().unwrap()[0];
        assert!((r - core::f32::consts::FRAC_PI_4).abs() < 1e-6);
    }

    #[test]
    fn sinh_cosh_at_zero() {
        let z = Tensor::from_vec([1usize], vec![0.0_f32]).unwrap();
        assert!(b().sinh(&z).unwrap().as_slice::<f32>().unwrap()[0].abs() < 1e-7);
        assert!((b().cosh(&z).unwrap().as_slice::<f32>().unwrap()[0] - 1.0).abs() < 1e-7);
    }

    // -------------------- P1.3 Activations completion --------------------

    #[test]
    fn erf_known_values() {
        // erf(0) = 0, erf(1) ≈ 0.8427, erf(∞) = 1
        assert!(super::erf_f32(0.0).abs() < 1e-7);
        assert!((super::erf_f32(1.0) - 0.842_701).abs() < 1e-5);
        assert!((super::erf_f32(2.0) - 0.995_322).abs() < 1e-5);
        // odd symmetry
        for &x in &[0.5_f32, 1.0, 1.5, 2.0, 3.0] {
            let pos = super::erf_f32(x);
            let neg = super::erf_f32(-x);
            assert!((pos + neg).abs() < 1e-6, "erf not odd at {x}");
        }
    }

    #[test]
    fn gelu_exact_at_zero() {
        let z = Tensor::from_vec([1usize], vec![0.0_f32]).unwrap();
        let r = b().gelu_exact(&z).unwrap().as_slice::<f32>().unwrap()[0];
        assert!(r.abs() < 1e-6);
    }

    #[test]
    fn gelu_exact_close_to_tanh_approx() {
        // tanh-approx and erf-exact differ by < 1e-3 over [-3, 3].
        let xs: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.1).collect();
        let t = Tensor::from_vec([xs.len()], xs.clone()).unwrap();
        let approx = b().gelu(&t).unwrap();
        let exact = b().gelu_exact(&t).unwrap();
        let av = approx.as_slice::<f32>().unwrap();
        let ev = exact.as_slice::<f32>().unwrap();
        for (i, (a, e)) in av.iter().zip(ev.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-3,
                "gelu approx/exact diverge at x={}: tanh={}, erf={}",
                xs[i],
                a,
                e
            );
        }
    }

    #[test]
    fn elu_negative_scaled_by_alpha() {
        let a = Tensor::from_vec([3usize], vec![-1.0_f32, 0.0, 1.0]).unwrap();
        let r = b().elu(&a, 1.0).unwrap();
        let v = r.as_slice::<f32>().unwrap();
        assert!((v[0] - (-0.6321)).abs() < 1e-3);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 1.0);
    }

    #[test]
    fn softplus_at_zero_is_log2() {
        let z = Tensor::from_vec([1usize], vec![0.0_f32]).unwrap();
        let r = b().softplus(&z, 1.0).unwrap().as_slice::<f32>().unwrap()[0];
        assert!((r - core::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn softplus_stable_at_extreme() {
        let big = Tensor::from_vec([1usize], vec![1e10_f32]).unwrap();
        let r = b().softplus(&big, 1.0).unwrap().as_slice::<f32>().unwrap()[0];
        assert!(r.is_finite());
        assert!((r - 1e10_f32).abs() < 1.0);
    }

    #[test]
    fn hardswish_ramp() {
        let a = Tensor::from_vec([4usize], vec![-3.0_f32, 0.0, 3.0, 6.0]).unwrap();
        let r = b()
            .hardswish(&a)
            .unwrap()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert!(r[0].abs() < 1e-6);
        assert!(r[1].abs() < 1e-6);
        assert!((r[2] - 3.0).abs() < 1e-6);
        assert!((r[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn hardtanh_clamps() {
        let a = Tensor::from_vec([4usize], vec![-2.0_f32, -1.0, 0.0, 2.0]).unwrap();
        let r = b()
            .hardtanh(&a, -1.0, 1.0)
            .unwrap()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert_eq!(r, vec![-1.0_f32, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn hardsigmoid_zero_three_neg_three() {
        let a = Tensor::from_vec([3usize], vec![-3.0_f32, 0.0, 3.0]).unwrap();
        let r = b()
            .hardsigmoid(&a)
            .unwrap()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert!(r[0].abs() < 1e-6);
        assert!((r[1] - 0.5).abs() < 1e-6);
        assert!((r[2] - 1.0).abs() < 1e-6);
    }

    // -------------------- Numerical-stability sanity --------------------

    #[test]
    fn sigmoid_stable_on_extreme_inputs() {
        let a = Tensor::from_vec([2usize], vec![-1e10_f32, 1e10]).unwrap();
        let r = b().sigmoid(&a).unwrap().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(r[0], 0.0);
        assert_eq!(r[1], 1.0);
    }

    // -------------------- Numerical-property tests --------------------

    #[test]
    fn sin_cos_pythagorean_identity_property() {
        let mut s: u64 = 0xCAFE1234;
        let mut data = Vec::with_capacity(50);
        for _ in 0..50 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let bits = (s ^ (s >> 32)) as u32;
            let v = ((bits as f32) / (u32::MAX as f32)) * 2.0 * core::f32::consts::PI
                - core::f32::consts::PI;
            data.push(v);
        }
        let t = Tensor::from_vec([data.len()], data).unwrap();
        let s_val = b().sin(&t).unwrap();
        let c_val = b().cos(&t).unwrap();
        let s2 = b().mul(&s_val, &s_val).unwrap();
        let c2 = b().mul(&c_val, &c_val).unwrap();
        let sum = b().add(&s2, &c2).unwrap();
        for &v in sum.as_slice::<f32>().unwrap() {
            assert!((v - 1.0).abs() < 1e-5, "sin²+cos² != 1: {}", v);
        }
    }

    // -------------------- Math edge cases (NaN/Inf semantics) --------------------

    #[test]
    fn log_zero_is_neg_inf() {
        let z = Tensor::from_vec([1usize], vec![0.0_f32]).unwrap();
        let r = b().log(&z).unwrap().as_slice::<f32>().unwrap()[0];
        assert!(
            r.is_infinite() && r.is_sign_negative(),
            "log(0) ≠ -Inf: {r}"
        );
    }

    #[test]
    fn log_neg_is_nan() {
        let n = Tensor::from_vec([1usize], vec![-1.0_f32]).unwrap();
        assert!(b().log(&n).unwrap().as_slice::<f32>().unwrap()[0].is_nan());
    }

    #[test]
    fn sqrt_neg_is_nan() {
        let n = Tensor::from_vec([1usize], vec![-1.0_f32]).unwrap();
        assert!(b().sqrt(&n).unwrap().as_slice::<f32>().unwrap()[0].is_nan());
    }

    #[test]
    fn exp_overflow_is_inf() {
        let big = Tensor::from_vec([1usize], vec![1000.0_f32]).unwrap();
        let r = b().exp(&big).unwrap().as_slice::<f32>().unwrap()[0];
        assert!(r.is_infinite());
    }

    #[test]
    fn nan_propagates_through_unary() {
        let n = Tensor::from_vec([1usize], vec![f32::NAN]).unwrap();
        for r in [
            b().sqrt(&n),
            b().exp(&n),
            b().log(&n),
            b().sin(&n),
            b().cos(&n),
            b().tan(&n),
            b().abs(&n),
            b().sigmoid(&n),
            b().tanh(&n),
            b().gelu(&n),
            b().silu(&n),
        ] {
            assert!(r.unwrap().as_slice::<f32>().unwrap()[0].is_nan());
        }
    }

    #[test]
    fn sigmoid_symmetric_property() {
        let mut s: u64 = 0x00C0_FFEE_BEEF;
        let mut data = Vec::with_capacity(50);
        for _ in 0..50 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let bits = (s ^ (s >> 32)) as u32;
            let v = ((bits as f32) / (u32::MAX as f32)) * 20.0 - 10.0;
            data.push(v);
        }
        let t = Tensor::from_vec([data.len()], data).unwrap();
        let neg_t = b().neg(&t).unwrap();
        let s1 = b().sigmoid(&t).unwrap();
        let s2 = b().sigmoid(&neg_t).unwrap();
        let sum = b().add(&s1, &s2).unwrap();
        for &v in sum.as_slice::<f32>().unwrap() {
            assert!((v - 1.0).abs() < 1e-6, "sigmoid(x)+sigmoid(-x) != 1: {}", v);
        }
    }

    // ----- mixed-precision parity (P3.1) ---------------------------------

    /// Build an `[m, k]` tensor in `dtype` from a Vec<f32> source.
    fn mk(shape: [usize; 2], data: Vec<f32>, dtype: Dtype) -> Tensor {
        let t = Tensor::from_vec(shape.to_vec(), data).unwrap();
        t.to_dtype(dtype)
    }

    fn cmp_via_f32(actual: &Tensor, expected: &Tensor, tol: f32) {
        let af = actual.to_dtype(Dtype::F32);
        let ef = expected.to_dtype(Dtype::F32);
        let av = af.as_slice::<f32>().unwrap();
        let ev = ef.as_slice::<f32>().unwrap();
        for (a, e) in av.iter().zip(ev) {
            let scale = e.abs().max(1.0);
            assert!(
                (a - e).abs() / scale < tol,
                "mismatch a={} e={} (rel tol {})",
                a,
                e,
                tol
            );
        }
    }

    #[test]
    fn add_bf16_parity_with_f32() {
        let a32 = mk([2, 3], vec![0.5, 1.0, -1.5, 2.5, 3.0, -4.5], Dtype::F32);
        let b32 = mk([2, 3], vec![0.25, -0.5, 0.75, -1.25, 2.0, 1.5], Dtype::F32);
        let a16 = a32.to_bf16();
        let b16 = b32.to_bf16();
        let r32 = b().add(&a32, &b32).unwrap();
        let r16 = b().add(&a16, &b16).unwrap();
        // bf16 has ~3 decimal digits of precision — 5e-3 relative tol.
        cmp_via_f32(&r16, &r32, 5e-3);
        assert_eq!(r16.dtype(), Dtype::BF16);
    }

    #[test]
    fn mul_bf16_parity_with_f32() {
        let a32 = mk(
            [4_usize, 4],
            (1..=16).map(|i| i as f32 * 0.1).collect(),
            Dtype::F32,
        );
        let b32 = mk(
            [4_usize, 4],
            (1..=16).map(|i| i as f32 * 0.05).collect(),
            Dtype::F32,
        );
        let a16 = a32.to_bf16();
        let b16 = b32.to_bf16();
        let r32 = b().mul(&a32, &b32).unwrap();
        let r16 = b().mul(&a16, &b16).unwrap();
        // bf16 mul rounds twice (each operand → 7-bit mantissa, then
        // product → 7-bit mantissa). 1.2e-2 covers worst-case rounding.
        cmp_via_f32(&r16, &r32, 1.2e-2);
    }

    #[test]
    fn matmul_bf16_parity_with_f32() {
        let a32 = mk(
            [4, 5],
            (0..20).map(|i| i as f32 * 0.1).collect(),
            Dtype::F32,
        );
        let b32 = mk(
            [5, 3],
            (0..15).map(|i| (i as f32 - 7.0) * 0.05).collect(),
            Dtype::F32,
        );
        let a16 = a32.to_bf16();
        let b16 = b32.to_bf16();
        let r32 = b().matmul(&a32, &b32).unwrap();
        let r16 = b().matmul(&a16, &b16).unwrap();
        // bf16 inputs cast to f32 → exact-ish computation → cast back.
        // Precision is dominated by the input round-trip.
        cmp_via_f32(&r16, &r32, 1e-2);
        assert_eq!(r16.shape(), [4, 3]);
        assert_eq!(r16.dtype(), Dtype::BF16);
    }

    // -------------------- A3 — autograd dispatch plumbing tests --------------------

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn assert_close_slice(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: len mismatch {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                close(*a, *e, tol),
                "{label}: idx {i} actual {a} expected {e} (tol {tol})"
            );
        }
    }

    #[test]
    fn bmm_simple_rank3() {
        // [B=2, M=2, K=3] @ [B=2, K=3, N=2] = [B=2, M=2, N=2]
        let a = Tensor::from_vec(
            [2usize, 2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, -1.0, 2.0, 1.0, 0.0],
        )
        .unwrap();
        let b_t = Tensor::from_vec(
            [2usize, 3, 2],
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0],
        )
        .unwrap();
        let out = b().bmm(&a, &b_t).unwrap();
        assert_eq!(out.shape(), [2, 2, 2]);
        let got = out.as_slice::<f32>().unwrap();
        // batch 0: [[1*1+2*0+3*1, 1*0+2*1+3*1], [4*1+5*0+6*1, 4*0+5*1+6*1]]
        //         = [[4, 5], [10, 11]]
        // batch 1: [[1*1+0*0+(-1)*(-1), 1*1+0*1+(-1)*0],
        //           [2*1+1*0+0*(-1),    2*1+1*1+0*0]]
        //         = [[2, 1], [2, 3]]
        assert_close_slice(
            got,
            &[4.0, 5.0, 10.0, 11.0, 2.0, 1.0, 2.0, 3.0],
            1e-6,
            "bmm",
        );
    }

    #[test]
    fn bmm_shape_mismatch_returns_err() {
        let a = Tensor::from_vec([2usize, 3, 4], vec![0.0; 24]).unwrap();
        let b_t = Tensor::from_vec([2usize, 5, 6], vec![0.0; 60]).unwrap();
        let err = b().bmm(&a, &b_t).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch { op: "bmm", .. }));
    }

    #[test]
    fn transpose_rank3_swap_last_two() {
        let a = Tensor::from_vec(
            [2usize, 2, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let out = b().transpose(&a, 1, 2).unwrap();
        assert_eq!(out.shape(), [2, 3, 2]);
        // batch 0 was rows [1,2,3] / [4,5,6] → cols (1,4),(2,5),(3,6)
        // contiguous layout: 1,4,2,5,3,6,7,10,8,11,9,12
        assert_close_slice(
            out.as_slice::<f32>().unwrap(),
            &[
                1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 7.0, 10.0, 8.0, 11.0, 9.0, 12.0,
            ],
            1e-6,
            "transpose",
        );
    }

    #[test]
    fn reshape_basic() {
        let a = Tensor::from_vec([2usize, 6], (0..12).map(|i| i as f32).collect()).unwrap();
        let out = b().reshape(&a, &[3, 4]).unwrap();
        assert_eq!(out.shape(), [3, 4]);
        // Data is the same flat sequence.
        assert_close_slice(
            out.as_slice::<f32>().unwrap(),
            &(0..12).map(|i| i as f32).collect::<Vec<_>>(),
            1e-6,
            "reshape",
        );
    }

    #[test]
    fn reshape_numel_mismatch_errs() {
        let a = Tensor::from_vec([2usize, 3], vec![0.0; 6]).unwrap();
        let err = b().reshape(&a, &[2, 4]).unwrap_err();
        assert!(matches!(
            err,
            BackendError::ShapeMismatch { op: "reshape", .. }
        ));
    }

    #[test]
    fn add_bias_broadcast_along_batch() {
        let x = Tensor::from_vec([3usize, 2], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let bias = Tensor::from_vec([2usize], vec![10.0_f32, 100.0]).unwrap();
        let out = b().add_bias(&x, &bias).unwrap();
        assert_eq!(out.shape(), [3, 2]);
        assert_close_slice(
            out.as_slice::<f32>().unwrap(),
            &[11.0, 102.0, 13.0, 104.0, 15.0, 106.0],
            1e-6,
            "add_bias",
        );
    }

    #[test]
    fn add_bias_rank_mismatch_errs() {
        let x = Tensor::from_vec([2usize, 2, 2], vec![0.0; 8]).unwrap(); // rank-3
        let bias = Tensor::from_vec([2usize], vec![0.0; 2]).unwrap();
        let err = b().add_bias(&x, &bias).unwrap_err();
        assert!(matches!(
            err,
            BackendError::ShapeMismatch { op: "add_bias", .. }
        ));
    }

    #[test]
    fn unbroadcast_to_identity() {
        // Same shape — no-op identity.
        let g = Tensor::from_vec([2usize, 3], (0..6).map(|i| i as f32).collect()).unwrap();
        let out = b().unbroadcast_to(&g, &[2, 3]).unwrap();
        assert_eq!(out.shape(), [2, 3]);
        assert_close_slice(
            out.as_slice::<f32>().unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            1e-6,
            "unbroadcast_id",
        );
    }

    #[test]
    fn unbroadcast_to_drop_leading_dim() {
        // grad [2, 3], target [3] → sum along axis 0, return [3].
        let g = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
        let out = b().unbroadcast_to(&g, &[3]).unwrap();
        assert_eq!(out.shape(), [3]);
        assert_close_slice(
            out.as_slice::<f32>().unwrap(),
            &[11.0, 22.0, 33.0],
            1e-6,
            "unbroadcast_drop_lead",
        );
    }

    #[test]
    fn unbroadcast_to_size_one_axis() {
        // grad [2, 3], target [2, 1] → sum along axis 1 keepdim.
        let g = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
        let out = b().unbroadcast_to(&g, &[2, 1]).unwrap();
        assert_eq!(out.shape(), [2, 1]);
        assert_close_slice(
            out.as_slice::<f32>().unwrap(),
            &[6.0, 60.0],
            1e-6,
            "unbroadcast_axis_one",
        );
    }

    #[test]
    fn unbroadcast_to_scalar_target() {
        // grad [2, 3], target [] (scalar via numel=1) → sum all.
        let g = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        // Empty target shape isn't representable, but [1] works and is what
        // the autograd dispatcher uses for scalar grad targets.
        let out = b().unbroadcast_to(&g, &[1]).unwrap();
        assert_eq!(out.shape(), [1]);
        assert_close_slice(
            out.as_slice::<f32>().unwrap(),
            &[21.0],
            1e-6,
            "unbroadcast_scalar",
        );
    }

    #[test]
    fn softmax_grad_matches_manual_formula() {
        // Build output = softmax(input). Pick an arbitrary grad and compare
        // backend.softmax_grad with the manual formula
        // d_input = output * (grad - sum(grad * output, dim, keepdim=true)).
        let input = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 0.5, 0.5, 0.5]).unwrap();
        let output = b().softmax(&input, 1).unwrap();
        let grad = Tensor::from_vec([2usize, 3], vec![0.1_f32, -0.2, 0.3, 1.0, 0.0, -1.0]).unwrap();
        let computed = b().softmax_grad(&grad, &output, 1).unwrap();

        // Manual: row 0
        let o = output.as_slice::<f32>().unwrap();
        let g = grad.as_slice::<f32>().unwrap();
        let mut expected = vec![0.0_f32; 6];
        for row in 0..2 {
            let mut sum_go = 0.0;
            for j in 0..3 {
                sum_go += g[row * 3 + j] * o[row * 3 + j];
            }
            for j in 0..3 {
                expected[row * 3 + j] = o[row * 3 + j] * (g[row * 3 + j] - sum_go);
            }
        }
        assert_close_slice(
            computed.as_slice::<f32>().unwrap(),
            &expected,
            1e-6,
            "softmax_grad",
        );
    }
}
