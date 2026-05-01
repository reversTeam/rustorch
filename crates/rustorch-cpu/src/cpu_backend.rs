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
            Dtype::F32 => matmul_naive::<f32>(lhs, rhs, m, k1, n),
            Dtype::F64 => matmul_naive::<f64>(lhs, rhs, m, k1, n),
            d => Err(BackendError::DtypeMismatch {
                op: "matmul",
                lhs: d,
                rhs: d,
            }),
        }
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

    fn cast(&self, src: &Tensor, target: Dtype) -> Result<Tensor, BackendError> {
        // Tensor::to_dtype already implements the full 8x8 matrix with
        // saturating semantics: Rust 1.45+ defines `f as i` as
        // saturating (NaN → 0, +Inf → MAX, -Inf → MIN), and Bool
        // conversions are explicit (any nonzero → true; true → 1.0,
        // false → 0.0). We just delegate.
        Ok(src.to_dtype(target))
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
        (d, _) => Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: d,
            rhs: d,
        }),
    }
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
}
