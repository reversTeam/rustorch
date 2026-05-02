//! `rustorch fork RUN_ID [--lr ...] [--optim ...]` — clone a previous
//! run's config + checkpoint and start a new run with overrides.
//!
//! v1 surfaces the contract: validate the run id format, print the
//! intended fork manifest. The actual catalog lookup + new-run
//! enqueue happens in plan P2.2 (Console HTTP API) once the run
//! database lands.

use crate::cli::{ForkArgs, LogFormat};
use crate::logging::{emit, Done, ErrorEvent, Started};

/// Entry point — returns the desired process exit code.
pub fn execute(args: &ForkArgs) -> i32 {
    let fmt = LogFormat::Pretty;

    if args.run_id.is_empty() {
        emit(
            fmt,
            &ErrorEvent {
                event: "error",
                op: "fork",
                message: "RUN_ID must not be empty",
            },
        );
        return 2;
    }
    if !is_valid_run_id(&args.run_id) {
        emit(
            fmt,
            &ErrorEvent {
                event: "error",
                op: "fork",
                message: &format!("invalid RUN_ID format: {}", args.run_id),
            },
        );
        return 2;
    }
    if let Some(lr) = args.lr {
        if !(lr.is_finite() && lr > 0.0) {
            emit(
                fmt,
                &ErrorEvent {
                    event: "error",
                    op: "fork",
                    message: &format!("invalid --lr override: {lr}"),
                },
            );
            return 2;
        }
    }

    emit(
        fmt,
        &Started {
            event: "started",
            op: "fork",
            context: &args.run_id,
        },
    );
    println!("[fork] base run     = {}", args.run_id);
    if let Some(lr) = args.lr {
        println!("[fork] override lr  = {lr}");
    }
    if let Some(opt) = args.optim.as_ref() {
        println!("[fork] override opt = {opt}");
    }
    println!("[fork] (catalog lookup wired in plan P2.2)");
    emit(
        fmt,
        &Done {
            event: "done",
            op: "fork",
            status: 0,
        },
    );
    0
}

/// Run-ids are `[a-zA-Z0-9_-]+` of length 1..=64 — enough to cover
/// any sane timestamp / hash format without drifting into XSS-risky
/// territory.
fn is_valid_run_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_run_id() {
        let args = ForkArgs {
            run_id: "".into(),
            lr: None,
            optim: None,
        };
        assert_eq!(execute(&args), 2);
    }

    #[test]
    fn rejects_invalid_run_id_chars() {
        let args = ForkArgs {
            run_id: "abc/def".into(),
            lr: None,
            optim: None,
        };
        assert_eq!(execute(&args), 2);
    }

    #[test]
    fn happy_path_with_overrides() {
        let args = ForkArgs {
            run_id: "abc-123".into(),
            lr: Some(1e-4),
            optim: Some("lion".into()),
        };
        assert_eq!(execute(&args), 0);
    }

    #[test]
    fn rejects_negative_lr_override() {
        let args = ForkArgs {
            run_id: "abc-123".into(),
            lr: Some(-1.0),
            optim: None,
        };
        assert_eq!(execute(&args), 2);
    }

    #[test]
    fn run_id_validator_table() {
        assert!(is_valid_run_id("abc"));
        assert!(is_valid_run_id("abc-123"));
        assert!(is_valid_run_id("a_b_c"));
        assert!(!is_valid_run_id(""));
        assert!(!is_valid_run_id("abc/def"));
        assert!(!is_valid_run_id("abc def"));
        // 65 char string → over the 64 cap.
        let big = "a".repeat(65);
        assert!(!is_valid_run_id(&big));
    }
}
