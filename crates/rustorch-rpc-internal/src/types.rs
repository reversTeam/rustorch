//! Wire types — same shape as `proto/runner.proto` but in plain
//! Rust + serde so they're usable without protoc.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub run_id: String,
    pub gpu_count: u32,
    pub runner_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub runner_id: String,
    pub session_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RunnerState {
    Starting,
    Running,
    Paused,
    Finished,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSample {
    pub step: i64,
    pub name: String,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogLine {
    pub ts_ms: i64,
    pub level: String,
    pub msg: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusUpdate {
    pub state: RunnerState,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSaved {
    pub path: String,
    pub step: i64,
    pub metrics_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTelemetry {
    pub gpu_id: u32,
    pub util: u32,
    pub vram_used_mb: u32,
    pub temp_c: u32,
    pub power_w: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerError {
    pub code: String,
    pub message: String,
    pub traceback: String,
}

/// Tagged event union — what the runner pushes on the upstream
/// channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunnerEvent {
    Metric(MetricSample),
    Log(LogLine),
    Status(StatusUpdate),
    Checkpoint(CheckpointSaved),
    Gpu(GpuTelemetry),
    Error(RunnerError),
}

/// Tagged command union — what the console sends downstream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControlCommand {
    Pause,
    Resume,
    Stop,
    Interrupt,
    SaveCheckpoint,
    UpdateConfig { overrides_json: String },
}
