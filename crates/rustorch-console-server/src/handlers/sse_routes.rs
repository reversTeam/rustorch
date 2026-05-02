//! HTTP entry points for the SSE topics. Each handler reads the
//! standard `Last-Event-ID` header (echoed by the browser on
//! reconnect) and passes it to the hub for replay of missed events.
//!
//! Topics implemented today:
//! * `runs.changed`         — every lifecycle transition
//! * `cluster.tick`         — heartbeat with GPU samples (1Hz NVML producer wires in P2.4)
//! * `run.:id.metric`       — per-run scalar metrics
//! * `run.:id.tail`         — per-run stdout tail
//! * `run.:id.checkpoint`  — per-run checkpoint saved
//! * `cargo.output`         — cargo run streaming (P2.5)
//!
//! All four core topics are wired; the dedicated producers for
//! `cluster.tick` (NVML) and `cargo.output` arrive in later slices.

use crate::sse::sse_for;
use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    response::sse::Sse,
    routing::get,
    Router,
};
use futures::Stream;
use std::convert::Infallible;

/// Read the standard `Last-Event-ID` header from a connect /
/// reconnect request. Browsers' `EventSource` sets this on every
/// reconnect attempt; manual `curl` clients can pass it too. We
/// parse it as a u64 (our hub IDs are monotonic counters).
fn last_event_id(headers: &HeaderMap) -> Option<u64> {
    headers
        .get("Last-Event-ID")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
}

/// `runs.changed` — emitted on every lifecycle transition.
pub async fn runs_changed(
    State(s): State<AppState>,
    headers: HeaderMap,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    sse_for(&s.hub, "runs.changed", last_event_id(&headers))
}

/// `cluster.tick` — heartbeat with GPU samples. The 1Hz NVML poller
/// will publish into this topic in a follow-up slice.
pub async fn cluster_tick(
    State(s): State<AppState>,
    headers: HeaderMap,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    sse_for(&s.hub, "cluster.tick", last_event_id(&headers))
}

/// `run.{id}.metric` — per-run metric stream.
pub async fn run_metric(
    State(s): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    let topic = format!("run.{id}.metric");
    sse_for(&s.hub, &topic, last_event_id(&headers))
}

/// `run.{id}.tail` — per-run stdout tail.
pub async fn run_tail(
    State(s): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    let topic = format!("run.{id}.tail");
    sse_for(&s.hub, &topic, last_event_id(&headers))
}

/// `run.{id}.checkpoint` — per-run checkpoint-saved events.
pub async fn run_checkpoint(
    State(s): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    let topic = format!("run.{id}.checkpoint");
    sse_for(&s.hub, &topic, last_event_id(&headers))
}

/// `cargo.output` — cargo run streaming (Editor "Run" button).
/// Producer wires in P2.5; route is exposed today so the frontend
/// can already subscribe.
pub async fn cargo_output(
    State(s): State<AppState>,
    headers: HeaderMap,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    sse_for(&s.hub, "cargo.output", last_event_id(&headers))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/sse/runs.changed", get(runs_changed))
        .route("/sse/cluster.tick", get(cluster_tick))
        .route("/sse/cargo.output", get(cargo_output))
        .route("/sse/runs/:id/metric", get(run_metric))
        .route("/sse/runs/:id/tail", get(run_tail))
        .route("/sse/runs/:id/checkpoint", get(run_checkpoint))
}
