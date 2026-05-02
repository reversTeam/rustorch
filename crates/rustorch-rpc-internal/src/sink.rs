//! `RpcRunnerSink` — implements `rustorch_log::Sink` and pushes
//! every `LogEvent` over the `RunnerClient` it wraps. Plug this
//! into `Logger::builder().with_sink(RpcRunnerSink::new(client))`
//! and every `log_metric()` call streams to the Console with no
//! extra wiring on the runner side.
//!
//! The bridge is one-way: rustorch-log → rustorch-rpc-internal.
//! We don't pull `LogEvent` into the rpc proto; instead we map our
//! variants to the closest `RunnerEvent` shape.

use crate::client::RunnerClient;
use crate::types::{CheckpointSaved as CheckpointSavedEvent, LogLine, MetricSample, RunnerEvent};
use rustorch_log::{LogEvent, Sink, SinkError};

/// Sink that translates `LogEvent` → `RunnerEvent` and pushes via
/// the gRPC stream (real or in-process).
pub struct RpcRunnerSink {
    client: RunnerClient,
}

impl RpcRunnerSink {
    pub fn new(client: RunnerClient) -> Self {
        Self { client }
    }
}

impl Sink for RpcRunnerSink {
    fn submit(&self, ev: &LogEvent) -> Result<(), SinkError> {
        let runner_event = match ev {
            LogEvent::Metric(m) => RunnerEvent::Metric(MetricSample {
                step: m.step,
                name: m.name.clone(),
                value: m.value,
            }),
            LogEvent::Event {
                name,
                level,
                payload,
                ts,
            } => RunnerEvent::Log(LogLine {
                ts_ms: ts.timestamp_millis(),
                level: format!("{level:?}").to_lowercase(),
                msg: format!("{name}: {payload}"),
            }),
            LogEvent::Histogram { .. } | LogEvent::Image { .. } | LogEvent::Embedding { .. } => {
                return Ok(())
            }, // skip — runner uploads via artifacts
            LogEvent::Hparams { .. } => return Ok(()), // sent via Register, not the stream
        };
        // We need a runtime to push the event because the client is
        // async. The Sink trait is sync — use tokio's current
        // handle if available; otherwise drop the event with a
        // warning (the Sink contract permits Err to be logged at
        // warn).
        let client = self.client.clone();
        match tokio::runtime::Handle::try_current() {
            Ok(h) => {
                h.spawn(async move {
                    client.send_event(runner_event).await;
                });
                Ok(())
            },
            Err(_) => Err(SinkError::Rejected(
                "RpcRunnerSink requires a running tokio runtime".into(),
            )),
        }
    }

    fn name(&self) -> &'static str {
        "rpc_runner"
    }
}

/// Helper to map a `Checkpoint` event onto the proto type — kept
/// distinct from the trait impl so callers can compose it.
pub fn checkpoint_event(path: String, step: i64, metrics_json: String) -> RunnerEvent {
    RunnerEvent::Checkpoint(CheckpointSavedEvent {
        path,
        step,
        metrics_json,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ClientConfig;
    use crate::registry::Registry;
    use crate::types::RegisterRequest;
    use rustorch_log::Logger;
    use std::time::Duration;

    fn req() -> RegisterRequest {
        RegisterRequest {
            run_id: "rid".into(),
            gpu_count: 1,
            runner_version: "0.0.1".into(),
        }
    }

    #[tokio::test]
    async fn sink_pushes_metric_via_rpc_client() {
        let registry = Registry::new();
        let client = RunnerClient::connect_in_process(registry.clone(), ClientConfig::default());
        let (resp, _session) = client.register(req());
        let mut rx = registry.take_events_rx(&resp.runner_id).unwrap();

        let logger = Logger::builder()
            .with_sink(RpcRunnerSink::new(client))
            .build();
        logger.metric("loss", 0.42, 7);

        // sink::submit spawns the actual send — give it a moment.
        let ev = tokio::time::timeout(Duration::from_millis(200), rx.recv())
            .await
            .unwrap()
            .unwrap();
        match ev {
            RunnerEvent::Metric(m) => {
                assert_eq!(m.name, "loss");
                assert_eq!(m.step, 7);
                assert!((m.value - 0.42).abs() < 1e-9);
            },
            _ => panic!("expected Metric"),
        }
    }

    #[tokio::test]
    async fn sink_translates_event_to_logline() {
        let registry = Registry::new();
        let client = RunnerClient::connect_in_process(registry.clone(), ClientConfig::default());
        let (resp, _session) = client.register(req());
        let mut rx = registry.take_events_rx(&resp.runner_id).unwrap();

        let logger = Logger::builder()
            .with_sink(RpcRunnerSink::new(client))
            .build();
        logger.event("epoch_complete", serde_json::json!({"acc": 0.91}));

        let ev = tokio::time::timeout(Duration::from_millis(200), rx.recv())
            .await
            .unwrap()
            .unwrap();
        match ev {
            RunnerEvent::Log(l) => {
                assert!(l.msg.contains("epoch_complete"));
                assert!(l.msg.contains("0.91"));
            },
            _ => panic!("expected LogLine"),
        }
    }
}
