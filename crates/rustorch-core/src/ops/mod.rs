//! Operations on `Tensor`.
//!
//! In the full design (RFC-0004 / RFC-0005), each op dispatches through a
//! generated match on `Storage`. In this prototype, ops live as free
//! functions that take and return `Tensor` directly.

mod add;
mod matmul;

pub use add::add;
pub use matmul::matmul;
