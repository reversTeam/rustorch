//! `rustorch deploy` — push a checkpoint to a serving endpoint.
//!
//! v1 validates inputs and prints the deployment manifest; the
//! actual HTTP POST happens in plan P2.6 (Inference & serving
//! runtime). Until then the command serves as the contract surface.

use crate::cli::{DeployArgs, LogFormat};
use crate::logging::{emit, Done, ErrorEvent, Started};

/// Entry point — returns the desired process exit code.
pub fn execute(args: &DeployArgs) -> i32 {
    let fmt = LogFormat::Pretty;

    if !args.checkpoint.exists() {
        emit(
            fmt,
            &ErrorEvent {
                event: "error",
                op: "deploy",
                message: &format!("checkpoint not found: {}", args.checkpoint.display()),
            },
        );
        return 2;
    }
    if !is_valid_url(&args.to) {
        emit(
            fmt,
            &ErrorEvent {
                event: "error",
                op: "deploy",
                message: &format!("invalid --to URL: {}", args.to),
            },
        );
        return 2;
    }

    emit(
        fmt,
        &Started {
            event: "started",
            op: "deploy",
            context: &format!("{} → {}", args.checkpoint.display(), args.to),
        },
    );
    // Real impl: POST {checkpoint, target_url, autoscale_cfg} to the
    // serving runtime API (plan P2.6). v1 stays no-op for the actual
    // HTTP call.
    println!(
        "[deploy] manifest:\n  checkpoint = {}\n  target     = {}\n  (HTTP POST wired in plan P2.6)",
        args.checkpoint.display(),
        args.to
    );
    emit(
        fmt,
        &Done {
            event: "done",
            op: "deploy",
            status: 0,
        },
    );
    0
}

/// Cheap URL sanity check — just verify the scheme is one of
/// `http://` / `https://`. A real validator would parse via `url`
/// crate; the CLI doesn't need that yet.
fn is_valid_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    #[test]
    fn rejects_missing_checkpoint() {
        let args = DeployArgs {
            checkpoint: PathBuf::from("/tmp/__rustorch_no_ckpt__.safetensors"),
            to: "https://api.example.com".into(),
        };
        assert_eq!(execute(&args), 2);
    }

    #[test]
    fn rejects_invalid_url() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "fake checkpoint").unwrap();
        let args = DeployArgs {
            checkpoint: f.path().to_path_buf(),
            to: "ftp://nope.example.com".into(),
        };
        assert_eq!(execute(&args), 2);
    }

    #[test]
    fn happy_path_with_real_file() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "fake checkpoint").unwrap();
        let args = DeployArgs {
            checkpoint: f.path().to_path_buf(),
            to: "https://api.example.com".into(),
        };
        assert_eq!(execute(&args), 0);
    }

    #[test]
    fn url_validator_table() {
        assert!(is_valid_url("http://x"));
        assert!(is_valid_url("https://x"));
        assert!(!is_valid_url("ftp://x"));
        assert!(!is_valid_url(""));
        assert!(!is_valid_url("example.com"));
    }
}
