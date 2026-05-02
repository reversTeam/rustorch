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
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Global SIGINT flag — set the first time the user hits Ctrl-C so the
/// fan-out loop can break early and reap any survivors. Wrapped in a
/// `OnceLock` so we install the handler at most once even when the
/// helper is called from multiple tests in the same process.
static SIGINT: OnceLock<Arc<AtomicBool>> = OnceLock::new();
/// Children spawned by the current `run` invocation, exposed to the
/// signal handler so it can `kill()` them when the user aborts.
static CHILDREN: OnceLock<Mutex<Vec<u32>>> = OnceLock::new();

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

    // ---- Install SIGINT handler ---------------------------------------
    // Installed lazily so unit tests can call `execute` repeatedly.
    let sigint = install_signal_handler();

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
        if sigint.load(Ordering::SeqCst) {
            // User aborted before we got to this rank — don't spawn it.
            last_status = 130; // POSIX SIGINT exit code
            break;
        }
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
        match cmd.spawn() {
            Ok(child) => {
                let pid = child.id();
                register_child(pid);
                let code = wait_or_interrupt(child, &sigint).unwrap_or(1);
                unregister_child(pid);
                if code != 0 {
                    last_status = code;
                    // Hard-fail policy: if a worker crashes and we still
                    // have ranks to spawn, kill any survivors so the
                    // user doesn't end up with orphaned RANK=0 stuck
                    // waiting on its peers. Single-GPU runs skip this
                    // (there are no siblings).
                    if args.gpus > 1 {
                        kill_all_children();
                        emit(
                            fmt,
                            &ErrorEvent {
                                event: "error",
                                op: "run",
                                message: &format!(
                                    "rank {local_rank} exited {code}; killed siblings"
                                ),
                            },
                        );
                        break;
                    }
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

/// Install (once) a Ctrl-C handler that flips the global `SIGINT`
/// flag and tries to terminate every still-tracked child. Returns the
/// shared flag so the caller can poll it between spawns.
fn install_signal_handler() -> Arc<AtomicBool> {
    let flag = SIGINT
        .get_or_init(|| {
            let f = Arc::new(AtomicBool::new(false));
            let f_clone = f.clone();
            // `ctrlc::set_handler` returns Err if a handler is already
            // installed (e.g. a parent test installed one). That's
            // fine — the existing handler still flips the same flag
            // because we go through `OnceLock`.
            let _ = ctrlc::set_handler(move || {
                f_clone.store(true, Ordering::SeqCst);
                kill_all_children();
            });
            f
        })
        .clone();
    // Reset between calls so a previous run's SIGINT doesn't bleed
    // into the next one (matters for the tests that call `execute`
    // back-to-back).
    flag.store(false, Ordering::SeqCst);
    flag
}

/// Wait for `child` to finish, polling the `sigint` flag every 100 ms
/// so the parent can react to Ctrl-C even when the worker is busy in
/// a long-running compile / training step. Returns the worker's exit
/// code (or 130 on interrupt).
fn wait_or_interrupt(mut child: Child, sigint: &Arc<AtomicBool>) -> std::io::Result<i32> {
    loop {
        if sigint.load(Ordering::SeqCst) {
            // Best-effort kill; if it already died we're fine.
            let _ = child.kill();
            let _ = child.wait();
            return Ok(130);
        }
        match child.try_wait()? {
            Some(status) => return Ok(status.code().unwrap_or(1)),
            None => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    }
}

fn register_child(pid: u32) {
    CHILDREN
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .push(pid);
}

fn unregister_child(pid: u32) {
    if let Some(m) = CHILDREN.get() {
        m.lock().unwrap().retain(|&p| p != pid);
    }
}

/// Kill every still-tracked child. Used by the SIGINT handler and the
/// "one worker died, drop the rest" path. Uses `kill(2)` directly on
/// Unix because we only have PIDs, not `Child` handles, by the time
/// the signal handler fires.
fn kill_all_children() {
    let Some(m) = CHILDREN.get() else { return };
    let pids: Vec<u32> = m.lock().unwrap().drain(..).collect();
    #[cfg(unix)]
    for pid in pids {
        // SIGTERM first so the worker has a chance to flush. Kernel
        // turns this into SIGKILL on stuck processes via the parent's
        // own exit anyway.
        unsafe {
            libc_kill(pid as i32, 15 /* SIGTERM */);
        }
    }
    #[cfg(not(unix))]
    for _pid in pids {
        // Windows: no portable PID-based kill in std without extra
        // crates. The Child handle in `wait_or_interrupt` already does
        // a `child.kill()` on the SIGINT path, so this is the rare
        // sibling-crash case where we accept that workers may linger
        // until the OS reaps them.
    }
}

// Tiny FFI to `kill(2)` so we don't pull in the whole `nix` crate
// just for one call. `libc::kill` would also work but adds a build-
// time dep we don't otherwise need.
#[cfg(unix)]
extern "C" {
    #[link_name = "kill"]
    fn libc_kill(pid: i32, sig: i32) -> i32;
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
