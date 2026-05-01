//! CPU backend kernels — split per category (shape, reduction, linalg, ...)
//! to keep each file focused and to give the codegen pipeline a stable
//! file-name target.

pub mod conv;
pub mod loss;
pub mod reduction;
pub mod shape_ops;
pub mod softmax;
