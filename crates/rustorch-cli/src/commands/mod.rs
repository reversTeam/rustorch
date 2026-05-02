//! Per-sub-command entry-points. Each `execute(args)` returns the
//! intended process exit code (0 = success, non-zero = error).

pub mod check;
pub mod deploy;
pub mod fork;
pub mod run;
pub mod sweep;
