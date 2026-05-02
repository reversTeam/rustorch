//! `rustorch run` — compile and run a training script with monitoring.
//!
//! v1 logic:
//! 1. Validate args (lr > 0, gpus > 0, etc.).
//! 2. Optionally merge a TOML config (`--config FILE.toml`).
//! 3. If `gpus == 1`, spawn `cargo run` directly with the rustorch
//!    env vars wired in. If `gpus > 1`, spawn N child processes with
//!    `WORLD_SIZE` / `RANK` set so distributed training works.
//! 4. Stream stdout / stderr as-is. The actual training loop lives
//!    in user code; rustorch's job here is the launcher.

use crate::cli::RunArgs;
use crate::config::load_config;
use crate::logging::{emit, Done, ErrorEvent, Started};
use std::process::Command;

/// Entry point — returns the desired process exit code.
pub fn execute(args: &RunArgs) -> i32 {
    let fmt = args.log_format;

    // ---- Argument validation ------------------------------------------
    if args.gpus == 0 {
        emit(
            fmt,
            &ErrorEvent {
                event: "error",
                op: "run",
                message: "--gpus must be > 0",
            },
        );
        return 2;
    }
    if !(args.lr.is_finite() && args.lr > 0.0) {
        emit(
            fmt,
            &ErrorEvent {
                event: "error",
                op: "run",
                message: "--lr must be finite and > 0",
            },
        );
        return 2;
    }
    if args.batch == 0 {
        emit(
            fmt,
            &ErrorEvent {
                event: "error",
                op: "run",
                message: "--batch must be > 0",
            },
        );
        return 2;
    }
    if !args.script.exists() {
        emit(
            fmt,
            &ErrorEvent {
                event: "error",
                op: "run",
                message: &format!("script not found: {}", args.script.display()),
            },
        );
        return 2;
    }

    // ---- Optional config merge ----------------------------------------
    let merged = match args.config.as_ref() {
        Some(p) => match load_config(p) {
            Ok(c) => Some(c),
            Err(e) => {
                emit(
                    fmt,
                    &ErrorEvent {
                        event: "error",
                        op: "run",
                        message: &format!("config load failed: {e}"),
                    },
                );
                return 2;
            },
        },
        None => None,
    };
    let world_size = args.world_size.unwrap_or(args.gpus);
    let _ = merged; // currently informational; full merge wiring is a follow-up

    emit(
        fmt,
        &Started {
            event: "started",
            op: "run",
            context: &args.script.display().to_string(),
        },
    );

    // ---- Spawn child processes ----------------------------------------
    // For gpus=1, we just run `cargo run` once (or rustc on standalone
    // files). For gpus>1, we fan out N children with WORLD_SIZE/RANK.
    let parent = args
        .script
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let cargo_toml = parent.join("Cargo.toml");

    let mut last_status = 0_i32;
    let local_ranks: Vec<u32> = (0..args.gpus).collect();
    for &local_rank in &local_ranks {
        let global_rank = args.rank.unwrap_or(0) * args.gpus + local_rank;
        let mut cmd = if cargo_toml.exists() {
            let mut c = Command::new("cargo");
            c.arg("run").arg("--manifest-path").arg(&cargo_toml);
            c
        } else {
            // Fallback: `rustc` to compile + run the standalone file.
            // Mostly useful for examples/tests.
            let mut c = Command::new("rustc");
            c.arg("--edition=2021").arg(&args.script);
            c
        };
        cmd.env("WORLD_SIZE", world_size.to_string())
            .env("RANK", global_rank.to_string())
            .env("LOCAL_RANK", local_rank.to_string())
            .env("MASTER_ADDR", "127.0.0.1")
            .env("MASTER_PORT", "29500")
            .env("RUSTORCH_LR", args.lr.to_string())
            .env("RUSTORCH_BATCH", args.batch.to_string())
            .env("RUSTORCH_EPOCHS", args.epochs.to_string())
            .env("RUSTORCH_AMP", format!("{:?}", args.amp))
            .env("RUSTORCH_OPTIM", &args.optim)
            .env("RUSTORCH_SEED", args.seed.to_string())
            .env("RUSTORCH_OUT", args.out.display().to_string());
        if let Some(resume) = args.resume.as_ref() {
            cmd.env("RUSTORCH_RESUME", resume.display().to_string());
        }
        match cmd.status() {
            Ok(s) => {
                let code = s.code().unwrap_or(1);
                if code != 0 {
                    last_status = code;
                }
            },
            Err(e) => {
                emit(
                    fmt,
                    &ErrorEvent {
                        event: "error",
                        op: "run",
                        message: &format!("rank {} spawn failed: {e}", local_rank),
                    },
                );
                last_status = 1;
            },
        }
    }

    emit(
        fmt,
        &Done {
            event: "done",
            op: "run",
            status: last_status,
        },
    );
    last_status
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{AmpMode, LogFormat, ProfileMode};
    use std::path::PathBuf;

    fn base_args() -> RunArgs {
        RunArgs {
            script: PathBuf::from("/tmp/__rustorch_doesnt_exist__.rs"),
            gpus: 1,
            batch: 32,
            lr: 1e-3,
            epochs: 1,
            amp: AmpMode::Fp32,
            optim: "adamw".into(),
            resume: None,
            seed: 0,
            out: PathBuf::from("runs/"),
            log_format: LogFormat::Pretty,
            profile: ProfileMode::Off,
            config: None,
            world_size: None,
            rank: None,
        }
    }

    #[test]
    fn rejects_zero_gpus() {
        let mut args = base_args();
        args.gpus = 0;
        assert_eq!(execute(&args), 2);
    }

    #[test]
    fn rejects_zero_batch() {
        let mut args = base_args();
        args.batch = 0;
        assert_eq!(execute(&args), 2);
    }

    #[test]
    fn rejects_non_positive_lr() {
        let mut args = base_args();
        args.lr = 0.0;
        assert_eq!(execute(&args), 2);
        args.lr = -1.0;
        assert_eq!(execute(&args), 2);
        args.lr = f32::NAN;
        assert_eq!(execute(&args), 2);
    }

    #[test]
    fn rejects_missing_script() {
        let args = base_args(); // script path doesn't exist
        assert_eq!(execute(&args), 2);
    }
}
