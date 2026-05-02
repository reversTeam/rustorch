//! `rustorch sweep` — queue N runs over a hyper-parameter grid.
//!
//! v1 emits a `[sweep]` summary line listing every (lr × batch)
//! combination that would be executed, plus a generated `sweep_id`
//! that runs share. The actual training queue is wired up in plan
//! P2.3 (Hyperparameter sweep system); this command is the CLI
//! surface that hands a sweep spec off to the queue.

use crate::cli::{LogFormat, SweepArgs, SweepStrategy};
use crate::logging::{emit, Done, ErrorEvent, Started};

/// Entry point — returns the desired process exit code.
pub fn execute(args: &SweepArgs) -> i32 {
    let fmt = LogFormat::Pretty;

    if args.lr.is_empty() && args.batch.is_empty() {
        emit(
            fmt,
            &ErrorEvent {
                event: "error",
                op: "sweep",
                message: "sweep needs at least one of --lr / --batch",
            },
        );
        return 2;
    }
    for &lr in &args.lr {
        if !(lr.is_finite() && lr > 0.0) {
            emit(
                fmt,
                &ErrorEvent {
                    event: "error",
                    op: "sweep",
                    message: &format!("invalid --lr value: {lr}"),
                },
            );
            return 2;
        }
    }
    for &b in &args.batch {
        if b == 0 {
            emit(
                fmt,
                &ErrorEvent {
                    event: "error",
                    op: "sweep",
                    message: "--batch values must be > 0",
                },
            );
            return 2;
        }
    }

    let combos = enumerate_combos(args);
    let sweep_id = mock_sweep_id();
    emit(
        fmt,
        &Started {
            event: "started",
            op: "sweep",
            context: &format!(
                "{} ({} combos via {:?})",
                args.script.display(),
                combos.len(),
                args.strategy
            ),
        },
    );
    println!("[sweep] sweep_id = {}", sweep_id);
    for (i, c) in combos.iter().enumerate() {
        println!(
            "[sweep] run {}/{}  lr={}  batch={}",
            i + 1,
            combos.len(),
            c.lr.map(|x| x.to_string()).unwrap_or_else(|| "—".into()),
            c.batch.map(|x| x.to_string()).unwrap_or_else(|| "—".into())
        );
    }
    emit(
        fmt,
        &Done {
            event: "done",
            op: "sweep",
            status: 0,
        },
    );
    0
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Combo {
    lr: Option<f32>,
    batch: Option<u32>,
}

/// Enumerate the (lr × batch) Cartesian product (Grid strategy) or
/// the first `trials` random samples (Random / others).
fn enumerate_combos(args: &SweepArgs) -> Vec<Combo> {
    match args.strategy {
        SweepStrategy::Grid => {
            // Cartesian product of every supplied list. Empty list ⇒
            // a single None placeholder so the other axis is still
            // expanded.
            let lrs: Vec<Option<f32>> = if args.lr.is_empty() {
                vec![None]
            } else {
                args.lr.iter().map(|x| Some(*x)).collect()
            };
            let batches: Vec<Option<u32>> = if args.batch.is_empty() {
                vec![None]
            } else {
                args.batch.iter().map(|x| Some(*x)).collect()
            };
            let mut out = Vec::with_capacity(lrs.len() * batches.len());
            for &lr in &lrs {
                for &b in &batches {
                    out.push(Combo { lr, batch: b });
                }
            }
            out
        },
        // For Random / Asha / Bayes the v1 CLI just enumerates the
        // grid up to `trials` items as a placeholder; the real
        // sampler lives in plan P2.3.
        _ => {
            let mut out = enumerate_combos(&SweepArgs {
                strategy: SweepStrategy::Grid,
                ..clone_sweep_args(args)
            });
            out.truncate(args.trials as usize);
            out
        },
    }
}

fn clone_sweep_args(args: &SweepArgs) -> SweepArgs {
    SweepArgs {
        script: args.script.clone(),
        lr: args.lr.clone(),
        batch: args.batch.clone(),
        strategy: args.strategy,
        trials: args.trials,
    }
}

/// Cheap deterministic sweep id from the env time. A real sweep
/// queue would assign one server-side; we just need something stable
/// for the CLI output.
fn mock_sweep_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("sw_{now:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn base_args() -> SweepArgs {
        SweepArgs {
            script: PathBuf::from("src/train.rs"),
            lr: vec![1e-4, 3e-4, 1e-3],
            batch: vec![64, 128],
            strategy: SweepStrategy::Grid,
            trials: 30,
        }
    }

    #[test]
    fn grid_yields_cartesian_product() {
        let args = base_args();
        let combos = enumerate_combos(&args);
        assert_eq!(combos.len(), 6); // 3 × 2
    }

    #[test]
    fn empty_axis_collapses_to_single_none() {
        let args = SweepArgs {
            script: PathBuf::from("src/train.rs"),
            lr: vec![1e-4, 3e-4, 1e-3],
            batch: vec![],
            strategy: SweepStrategy::Grid,
            trials: 30,
        };
        let combos = enumerate_combos(&args);
        assert_eq!(combos.len(), 3);
        assert!(combos.iter().all(|c| c.batch.is_none()));
    }

    #[test]
    fn rejects_empty_args() {
        let args = SweepArgs {
            script: PathBuf::from("src/train.rs"),
            lr: vec![],
            batch: vec![],
            strategy: SweepStrategy::Grid,
            trials: 30,
        };
        assert_eq!(execute(&args), 2);
    }

    #[test]
    fn rejects_negative_lr() {
        let mut args = base_args();
        args.lr = vec![-1.0];
        assert_eq!(execute(&args), 2);
    }

    #[test]
    fn rejects_zero_batch() {
        let mut args = base_args();
        args.batch = vec![0];
        assert_eq!(execute(&args), 2);
    }

    #[test]
    fn random_strategy_truncates_to_trials() {
        let args = SweepArgs {
            strategy: SweepStrategy::Random,
            trials: 4,
            ..base_args()
        };
        let combos = enumerate_combos(&args);
        assert_eq!(combos.len(), 4);
    }
}
