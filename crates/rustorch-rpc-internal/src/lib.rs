//! Internal RPC layer between the Console backend and the Runners.
//!
//! Two layers ship today:
//!
//! 1. **Wire schema** — `proto/runner.proto`. The single source of
//!    truth for what events a runner can push and what commands the
//!    console can send. Generated stubs come up behind feature
//!    `grpc` (needs `tonic` + `protoc` on the toolchain).
//!
//! 2. **In-process implementation** — `Registry` + `RunnerSession`.
//!    Same shape as the gRPC contract but all in-Rust, useful for
//!    unit tests of the Console state machine and for embedded
//!    deployments where the Console and Runner live in the same
//!    process.
//!
//! When the gRPC feature lands, the in-process implementation stays
//! as the test backend; production code switches to tonic with the
//! same trait shape so callers don't change.

mod client;
mod registry;
mod sink;
mod types;

pub use client::{ClientConfig, ClientStats, RunnerClient};
pub use registry::{Registry, RegistryError, RunnerSession};
pub use sink::{checkpoint_event, RpcRunnerSink};
pub use types::{
    CheckpointSaved, ControlCommand, GpuTelemetry, LogLine, MetricSample, RegisterRequest,
    RegisterResponse, RunnerError, RunnerEvent, RunnerState, StatusUpdate,
};

#[cfg(feature = "grpc")]
pub mod grpc {
    //! Placeholder for the tonic-generated stubs. Activated when the
    //! `grpc` feature is on; today the module is empty so callers
    //! compile-test the feature gate.
}
