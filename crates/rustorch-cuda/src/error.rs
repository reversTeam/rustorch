//! Unified CUDA error type — maps driver / runtime / cuBLAS / cuDNN /
//! cuSPARSE / cuRAND status codes onto a single `Result`-friendly
//! enum. Each variant carries the originating subsystem code so
//! callers can emit actionable diagnostics.

/// Errors raised by any rustorch-cuda operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    /// No CUDA-capable device visible to this process. Returned when
    /// `--features cuda` is OFF, or when `cuInit(0)` fails because
    /// the driver isn't loaded.
    NoDeviceFound,
    /// CUDA driver / runtime status code.
    Driver {
        /// Raw status code (CUresult or cudaError_t).
        code: i32,
        /// Source location of the failing call.
        location: &'static str,
    },
    /// cuBLAS status code.
    CublasStatus {
        /// Raw `cublasStatus_t`.
        code: i32,
        /// Source location.
        location: &'static str,
    },
    /// cuDNN status code.
    CudnnStatus {
        /// Raw `cudnnStatus_t`.
        code: i32,
        /// Source location.
        location: &'static str,
    },
    /// cuSPARSE status code.
    CusparseStatus {
        /// Raw `cusparseStatus_t`.
        code: i32,
        /// Source location.
        location: &'static str,
    },
    /// cuRAND status code.
    CurandStatus {
        /// Raw `curandStatus_t`.
        code: i32,
        /// Source location.
        location: &'static str,
    },
    /// Out of memory on the device.
    Oom {
        /// Bytes requested.
        requested: u64,
    },
    /// Operation requested isn't supported by this build (e.g. the
    /// `cuda` feature flag is off, or the descriptor combo isn't
    /// implemented yet).
    Unsupported {
        /// Free-form diagnostic.
        msg: String,
    },
    /// Unknown / unmapped status code.
    Unknown(i32),
}

impl core::fmt::Display for CudaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CudaError::NoDeviceFound => write!(f, "no CUDA-capable device found"),
            CudaError::Driver { code, location } => {
                write!(f, "CUDA driver error {code} at {location}")
            },
            CudaError::CublasStatus { code, location } => {
                write!(f, "cuBLAS status {code} at {location}")
            },
            CudaError::CudnnStatus { code, location } => {
                write!(f, "cuDNN status {code} at {location}")
            },
            CudaError::CusparseStatus { code, location } => {
                write!(f, "cuSPARSE status {code} at {location}")
            },
            CudaError::CurandStatus { code, location } => {
                write!(f, "cuRAND status {code} at {location}")
            },
            CudaError::Oom { requested } => write!(f, "device OOM: requested {requested} bytes"),
            CudaError::Unsupported { msg } => write!(f, "unsupported: {msg}"),
            CudaError::Unknown(code) => write!(f, "unknown CUDA status {code}"),
        }
    }
}

impl std::error::Error for CudaError {}

/// Macro: `check_cuda!(cuFooBar(args), "context::init")` — wraps a raw
/// FFI call, mapping a non-zero status to `CudaError::Driver` with
/// the supplied location string. Defined for both feature-on and
/// feature-off paths so call sites compile uniformly.
#[macro_export]
macro_rules! check_cuda {
    ($call:expr, $loc:literal) => {{
        let status: i32 = $call as i32;
        if status == 0 {
            Ok::<_, $crate::error::CudaError>(())
        } else {
            Err($crate::error::CudaError::Driver {
                code: status,
                location: $loc,
            })
        }
    }};
}

/// Macro: `check_cublas!(cublasGemmEx(...), "cublas::gemm")`.
#[macro_export]
macro_rules! check_cublas {
    ($call:expr, $loc:literal) => {{
        let status: i32 = $call as i32;
        if status == 0 {
            Ok::<_, $crate::error::CudaError>(())
        } else {
            Err($crate::error::CudaError::CublasStatus {
                code: status,
                location: $loc,
            })
        }
    }};
}

/// Macro: `check_cudnn!(cudnnConvolutionForward(...), "cudnn::conv2d")`.
#[macro_export]
macro_rules! check_cudnn {
    ($call:expr, $loc:literal) => {{
        let status: i32 = $call as i32;
        if status == 0 {
            Ok::<_, $crate::error::CudaError>(())
        } else {
            Err($crate::error::CudaError::CudnnStatus {
                code: status,
                location: $loc,
            })
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_device_found_displays_humanly() {
        let s = format!("{}", CudaError::NoDeviceFound);
        assert!(s.contains("no CUDA-capable device"));
    }

    #[test]
    fn driver_error_includes_location() {
        let e = CudaError::Driver {
            code: 42,
            location: "context::init",
        };
        let s = format!("{e}");
        assert!(s.contains("42"));
        assert!(s.contains("context::init"));
    }

    #[test]
    fn unknown_code_does_not_panic() {
        let e = CudaError::Unknown(999_999);
        let _ = format!("{e}");
    }

    #[test]
    fn check_cuda_macro_passes_zero_status() {
        let r: Result<_, CudaError> = check_cuda!(0i32, "test::ok");
        assert!(r.is_ok());
    }

    #[test]
    fn check_cuda_macro_returns_err_on_nonzero() {
        let r: Result<_, CudaError> = check_cuda!(7i32, "test::fail");
        match r {
            Err(CudaError::Driver { code: 7, location }) => assert_eq!(location, "test::fail"),
            _ => panic!("expected Driver error"),
        }
    }

    #[test]
    fn check_cublas_macro_routes_to_cublas_variant() {
        let r: Result<_, CudaError> = check_cublas!(13i32, "cublas::gemm");
        assert!(matches!(r, Err(CudaError::CublasStatus { code: 13, .. })));
    }

    #[test]
    fn check_cudnn_macro_routes_to_cudnn_variant() {
        let r: Result<_, CudaError> = check_cudnn!(99i32, "cudnn::conv");
        assert!(matches!(r, Err(CudaError::CudnnStatus { code: 99, .. })));
    }

    #[test]
    fn oom_includes_requested_bytes() {
        let e = CudaError::Oom { requested: 1024 };
        assert!(format!("{e}").contains("1024"));
    }
}
