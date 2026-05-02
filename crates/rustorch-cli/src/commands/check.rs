//! `rustorch check PATH.rs` — run `cargo check` on the given script
//! and surface diagnostics. Returns the cargo exit code (0 = clean,
//! non-zero = compile error).

use crate::cli::{CheckArgs, LogFormat};
use crate::logging::{emit, Done, ErrorEvent, Started};
use std::process::Command;

/// Entry point. Spawns `cargo check` and waits for it.
pub fn execute(args: &CheckArgs) -> i32 {
    let fmt = LogFormat::Pretty;
    if !args.script.exists() {
        emit(
            fmt,
            &ErrorEvent {
                event: "error",
                op: "check",
                message: &format!("script not found: {}", args.script.display()),
            },
        );
        return 2;
    }
    emit(
        fmt,
        &Started {
            event: "started",
            op: "check",
            context: &args.script.display().to_string(),
        },
    );

    // Run `cargo check` in the script's parent directory if it lives
    // in a Cargo project; otherwise just check the file via `rustc`
    // (best-effort, won't resolve crate deps).
    let parent = args
        .script
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let cargo_toml = parent.join("Cargo.toml");
    let status = if cargo_toml.exists() {
        Command::new("cargo")
            .arg("check")
            .arg("--manifest-path")
            .arg(&cargo_toml)
            .status()
    } else {
        // Standalone .rs file — `rustc --edition=2021 --emit=metadata`
        // is the lightest no-deps check we can run.
        Command::new("rustc")
            .arg("--edition=2021")
            .arg("--emit=metadata")
            .arg("--out-dir=/tmp")
            .arg(&args.script)
            .status()
    };
    let code = match status {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            emit(
                fmt,
                &ErrorEvent {
                    event: "error",
                    op: "check",
                    message: &format!("failed to run cargo: {e}"),
                },
            );
            return 1;
        },
    };
    emit(
        fmt,
        &Done {
            event: "done",
            op: "check",
            status: code,
        },
    );
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_script_returns_2() {
        let args = CheckArgs {
            script: std::path::PathBuf::from("/tmp/this/does/not/exist.rs"),
        };
        assert_eq!(execute(&args), 2);
    }
}
