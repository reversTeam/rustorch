//! TOML config file loader for `rustorch run --config FILE.toml`.
//!
//! Lets users record the full set of run flags in a versioned file
//! and load them with one CLI flag. CLI flags **override** config
//! values when both are supplied.
//!
//! Schema:
//! ```toml
//! [run]
//! gpus       = 4
//! batch      = 256
//! lr         = 0.0003
//! epochs     = 10
//! amp        = "bf16"
//! optim      = "adamw"
//! seed       = 0
//! out        = "runs/"
//! log_format = "pretty"
//! profile    = "off"
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Subset of [`crate::cli::RunArgs`] fields that may live in a TOML
/// config file. Any field omitted falls back to the CLI default.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    /// Number of GPUs to spawn (DDP world size on a single node).
    pub gpus: Option<u32>,
    /// Per-GPU mini-batch size.
    pub batch: Option<u32>,
    /// Initial learning rate.
    pub lr: Option<f32>,
    /// Epochs to train.
    pub epochs: Option<u32>,
    /// Mixed-precision policy as a string (`fp32` / `fp16` / `bf16`).
    pub amp: Option<String>,
    /// Optimizer name.
    pub optim: Option<String>,
    /// Random seed.
    pub seed: Option<u64>,
    /// Output directory for checkpoints / logs.
    pub out: Option<String>,
    /// Output format (`pretty` / `json`).
    pub log_format: Option<String>,
    /// Profiling mode (`off` / `sampling` / `full`).
    pub profile: Option<String>,
}

/// Top-level config — sectioned by sub-command name.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Config {
    /// `[run]` section.
    #[serde(default)]
    pub run: RunConfig,
}

/// Load `path` as TOML and parse into a [`Config`].
pub fn load_config(path: &Path) -> Result<Config, ConfigError> {
    let s = std::fs::read_to_string(path).map_err(|e| ConfigError::Read(e.to_string()))?;
    toml::from_str::<Config>(&s).map_err(|e| ConfigError::Parse(e.to_string()))
}

/// Errors raised by the config loader.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Filesystem read failed (file missing, permission denied, …).
    #[error("config read error: {0}")]
    Read(String),
    /// TOML parse error (syntax, type mismatch, …).
    #[error("config parse error: {0}")]
    Parse(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_minimal_config() {
        let src = r#"
            [run]
            gpus = 4
            batch = 256
            lr = 0.0003
            amp = "bf16"
        "#;
        let cfg: Config = toml::from_str(src).unwrap();
        assert_eq!(cfg.run.gpus, Some(4));
        assert_eq!(cfg.run.batch, Some(256));
        assert_eq!(cfg.run.amp.as_deref(), Some("bf16"));
    }

    #[test]
    fn missing_section_yields_defaults() {
        let cfg: Config = toml::from_str("").unwrap();
        assert!(cfg.run.gpus.is_none());
    }

    #[test]
    fn round_trip_via_disk() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "[run]\ngpus = 2\nlr = 0.001").unwrap();
        let cfg = load_config(f.path()).unwrap();
        assert_eq!(cfg.run.gpus, Some(2));
        assert_eq!(cfg.run.lr, Some(0.001));
    }

    #[test]
    fn rejects_invalid_toml() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "this is not = valid: toml :::").unwrap();
        let err = load_config(f.path()).unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)));
    }

    #[test]
    fn rejects_missing_file() {
        let path = std::path::Path::new("/tmp/this/does/not/exist/rustorch.toml");
        let err = load_config(path).unwrap_err();
        assert!(matches!(err, ConfigError::Read(_)));
    }
}
