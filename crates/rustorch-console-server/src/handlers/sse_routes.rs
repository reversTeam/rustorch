//! HTTP entry points for the SSE topics. Each route just resolves
//! its topic name and delegates to the hub. Real filtering (per-run
//! topic) lives in the topic name (`run.{id}.metric`).

use crate::sse::sse_for;
use crate::state::AppState;
use axum::{
    extract::{Path, State},
    response::sse::Sse,
    routing::get,
    Router,
};
use futures::Stream;
use std::convert::Infallible;

/// `runs.changed` — emitted on every lifecycle transition (create,
/// start, pause, resume, stop, …).
pub async fn runs_changed(
    State(s): State<AppState>,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    sse_for(&s.hub, "runs.changed")
}

/// `cluster.tick` — heartbeat with GPU samples. The producer side
/// (1Hz NVML poll) lands in P2.4; today the topic is wired but no
/// publisher runs unless something explicitly calls `hub.publish`.
pub async fn cluster_tick(
    State(s): State<AppState>,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    sse_for(&s.hub, "cluster.tick")
}

/// `run.{id}.metric` — per-run metric stream.
pub async fn run_metric(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    let topic = format!("run.{id}.metric");
    sse_for(&s.hub, &topic)
}

/// `run.{id}.tail` — per-run stdout tail.
pub async fn run_tail(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    let topic = format!("run.{id}.tail");
    sse_for(&s.hub, &topic)
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/sse/runs.changed", get(runs_changed))
        .route("/sse/cluster.tick", get(cluster_tick))
        .route("/sse/runs/:id/metric", get(run_metric))
        .route("/sse/runs/:id/tail", get(run_tail))
}
