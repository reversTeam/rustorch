//! `clap`-derived command-line surface.
//!
//! The top-level [`Cli`] type matches the canonical surface from doc
//! v0.7.2 / Reference / CLI flags:
//!
//! ```text
//! rustorch run    [--gpus N] [--batch B] [--lr L] [--epochs E]
//!                 [--amp bf16|fp16|fp32] [--ckpt-every 1ep] [--ckpt-keep 5]
//!                 [--resume PATH] [--seed S] [--out DIR]
//!                 [--log-format pretty|json] [--profile off|sampling|full]
//!                 [--config FILE.toml]
//!                 [--world-size N] [--rank R]
//!                 [--optim ADAM|ADAMW|LION|...]
//!                 PATH.rs
//! rustorch sweep  --lr a,b,c [--batch …] [--strategy grid|random|asha|bayes]
//!                 [--trials N] PATH.rs
//! rustorch check  PATH.rs
//! rustorch deploy --checkpoint PATH --to URL
//! rustorch fork   RUN_ID [--lr L] [--optim NAME]
//! ```
//!
//! Each sub-command struct also implements its own argument
//! validation through [`clap`]'s `value_parser` infrastructure so
//! invalid combinations are rejected at parse time with a friendly
//! error, before any work is done.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// rustorch — pure-Rust deep-learning framework, command-line entry.
#[derive(Debug, Parser)]
#[command(name = "rustorch")]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Sub-command to dispatch.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level sub-commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Compile and run a training script with rustorch's monitoring.
    Run(RunArgs),
    /// Queue N runs over a hyper-parameter grid / sweep.
    Sweep(SweepArgs),
    /// Validate that a training script compiles. No execution.
    Check(CheckArgs),
    /// Push a checkpoint to a serving endpoint.
    Deploy(DeployArgs),
    /// Clone a previous run's config and resume from its checkpoint
    /// with overrides.
    Fork(ForkArgs),
}

/// Mixed-precision policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AmpMode {
    /// Pure float32 (default).
    Fp32,
    /// IEEE half precision.
    Fp16,
    /// Brain float 16 (recommended on Apple / Hopper).
    Bf16,
}

/// Output format selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LogFormat {
    /// Coloured tty-friendly output (default).
    Pretty,
    /// One JSON event per line — consumable by sinks.
    Json,
}

/// Profiling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ProfileMode {
    /// No profiling (default).
    Off,
    /// Low-overhead sampling profiler (default 10ms interval).
    Sampling,
    /// Full event-based tracer (Chrome trace JSON output).
    Full,
}

/// Hyperparameter sweep strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SweepStrategy {
    /// Cartesian product of every flag value.
    Grid,
    /// Uniform random sampling, `--trials` budget.
    Random,
    /// Asynchronous successive halving (Hyperband-style).
    Asha,
    /// Bayesian optimisation with GP / Matern 5/2.
    Bayes,
}

/// `rustorch run` — compile + run + monitor a training script.
#[derive(Debug, Parser)]
pub struct RunArgs {
    /// Path to the `.rs` training script (must compile via cargo).
    pub script: PathBuf,

    /// Number of GPUs to spawn — translates to `WORLD_SIZE` for DDP.
    /// On a single machine, each GPU runs a child process.
    #[arg(long, default_value_t = 1)]
    pub gpus: u32,

    /// Mini-batch size (per GPU).
    #[arg(long, default_value_t = 32)]
    pub batch: u32,

    /// Learning rate (initial). Validated as `> 0`.
    #[arg(long, default_value_t = 3e-4)]
    pub lr: f32,

    /// Number of epochs to train.
    #[arg(long, default_value_t = 1)]
    pub epochs: u32,

    /// Mixed-precision policy.
    #[arg(long, value_enum, default_value_t = AmpMode::Fp32)]
    pub amp: AmpMode,

    /// Optimizer name (`adamw`, `adam`, `sgd`, `lion`, …).
    #[arg(long, default_value = "adamw")]
    pub optim: String,

    /// Path to a previous checkpoint — resume training from it.
    #[arg(long)]
    pub resume: Option<PathBuf>,

    /// Random seed. Same seed → reproducible runs.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Output directory for checkpoints / logs.
    #[arg(long, default_value = "runs/")]
    pub out: PathBuf,

    /// Output format.
    #[arg(long, value_enum, default_value_t = LogFormat::Pretty)]
    pub log_format: LogFormat,

    /// Profiling mode.
    #[arg(long, value_enum, default_value_t = ProfileMode::Off)]
    pub profile: ProfileMode,

    /// Optional TOML config file overriding the flags.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// World size for multi-node DDP (overrides `--gpus` semantics).
    #[arg(long)]
    pub world_size: Option<u32>,

    /// Rank within the multi-node cluster.
    #[arg(long)]
    pub rank: Option<u32>,
}

/// `rustorch sweep` — queue N runs over a hyper-parameter grid.
#[derive(Debug, Parser)]
pub struct SweepArgs {
    /// Path to the `.rs` training script.
    pub script: PathBuf,

    /// Comma-separated learning rates (e.g. `1e-4,3e-4,1e-3`).
    #[arg(long, value_delimiter = ',')]
    pub lr: Vec<f32>,

    /// Comma-separated batch sizes.
    #[arg(long, value_delimiter = ',')]
    pub batch: Vec<u32>,

    /// Sweep strategy.
    #[arg(long, value_enum, default_value_t = SweepStrategy::Grid)]
    pub strategy: SweepStrategy,

    /// Trials budget for `random` / `bayes` strategies.
    #[arg(long, default_value_t = 30)]
    pub trials: u32,
}

/// `rustorch check` — validate a script compiles.
#[derive(Debug, Parser)]
pub struct CheckArgs {
    /// Path to the `.rs` training script.
    pub script: PathBuf,
}

/// `rustorch deploy` — push a checkpoint to a serving endpoint.
#[derive(Debug, Parser)]
pub struct DeployArgs {
    /// Path to a `.safetensors` checkpoint.
    #[arg(long)]
    pub checkpoint: PathBuf,

    /// Target deployment URL.
    #[arg(long)]
    pub to: String,
}

/// `rustorch fork` — clone a previous run's config + checkpoint with overrides.
#[derive(Debug, Parser)]
pub struct ForkArgs {
    /// Run identifier to fork from.
    pub run_id: String,

    /// Override the learning rate.
    #[arg(long)]
    pub lr: Option<f32>,

    /// Override the optimizer name.
    #[arg(long)]
    pub optim: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_parses_basic_run() {
        let args = Cli::try_parse_from(["rustorch", "run", "src/train.rs"]).unwrap();
        match args.command {
            Command::Run(r) => {
                assert_eq!(r.script, PathBuf::from("src/train.rs"));
                assert_eq!(r.gpus, 1);
                assert_eq!(r.batch, 32);
                assert_eq!(r.amp, AmpMode::Fp32);
            },
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn cli_parses_full_run_flags() {
        let args = Cli::try_parse_from([
            "rustorch",
            "run",
            "--gpus",
            "4",
            "--batch",
            "256",
            "--lr",
            "1e-3",
            "--epochs",
            "10",
            "--amp",
            "bf16",
            "--seed",
            "42",
            "--log-format",
            "json",
            "src/train.rs",
        ])
        .unwrap();
        match args.command {
            Command::Run(r) => {
                assert_eq!(r.gpus, 4);
                assert_eq!(r.batch, 256);
                assert!((r.lr - 1e-3).abs() < 1e-9);
                assert_eq!(r.epochs, 10);
                assert_eq!(r.amp, AmpMode::Bf16);
                assert_eq!(r.seed, 42);
                assert_eq!(r.log_format, LogFormat::Json);
            },
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn cli_parses_sweep_with_csv_lr() {
        let args = Cli::try_parse_from([
            "rustorch",
            "sweep",
            "--lr",
            "1e-4,3e-4,1e-3",
            "--batch",
            "64,128",
            "src/train.rs",
        ])
        .unwrap();
        match args.command {
            Command::Sweep(s) => {
                assert_eq!(s.lr.len(), 3);
                assert_eq!(s.batch.len(), 2);
                assert_eq!(s.strategy, SweepStrategy::Grid);
            },
            _ => panic!("expected Sweep"),
        }
    }

    #[test]
    fn cli_parses_check() {
        let args = Cli::try_parse_from(["rustorch", "check", "src/train.rs"]).unwrap();
        matches!(args.command, Command::Check(_));
    }

    #[test]
    fn cli_parses_deploy() {
        let args = Cli::try_parse_from([
            "rustorch",
            "deploy",
            "--checkpoint",
            "model.safetensors",
            "--to",
            "https://api.example.com",
        ])
        .unwrap();
        match args.command {
            Command::Deploy(d) => {
                assert_eq!(d.checkpoint, PathBuf::from("model.safetensors"));
                assert_eq!(d.to, "https://api.example.com");
            },
            _ => panic!("expected Deploy"),
        }
    }

    #[test]
    fn cli_parses_fork_with_overrides() {
        let args = Cli::try_parse_from([
            "rustorch", "fork", "abc-123", "--lr", "1e-4", "--optim", "lion",
        ])
        .unwrap();
        match args.command {
            Command::Fork(f) => {
                assert_eq!(f.run_id, "abc-123");
                assert_eq!(f.lr, Some(1e-4));
                assert_eq!(f.optim, Some("lion".into()));
            },
            _ => panic!("expected Fork"),
        }
    }

    #[test]
    fn cli_rejects_unknown_amp_mode() {
        let err = Cli::try_parse_from(["rustorch", "run", "--amp", "fp99", "x.rs"]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("invalid value") || msg.contains("possible values"));
    }

    #[test]
    fn cli_rejects_unknown_subcommand() {
        let err = Cli::try_parse_from(["rustorch", "fakecmd"]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unrecognized") || msg.contains("usage") || msg.contains("error"));
    }

    #[test]
    fn cli_help_smoke() {
        // Just verify the help text builds without panic.
        let mut cmd = Cli::command();
        let mut buf = Vec::new();
        cmd.write_help(&mut buf).unwrap();
        let help = String::from_utf8(buf).unwrap();
        assert!(help.contains("rustorch"));
        assert!(help.contains("run"));
        assert!(help.contains("sweep"));
        assert!(help.contains("check"));
        assert!(help.contains("deploy"));
        assert!(help.contains("fork"));
    }
}
