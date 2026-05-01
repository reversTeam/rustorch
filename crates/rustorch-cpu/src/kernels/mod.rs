//! CPU backend kernels — split per category (shape, reduction, linalg, ...)
//! to keep each file focused and to give the codegen pipeline a stable
//! file-name target.

pub mod conv;
pub mod ctc;
pub mod einsum;
pub mod linalg;
pub mod loss;
pub mod norm;
pub mod pool;
pub mod reduction;
pub mod shape_ops;
pub mod softmax;
