//! Runs lifecycle endpoints.
//!
//! The state machine is intentionally small for v1:
//!   queued    → running       (start)
//!   running   → paused        (pause)
//!   paused    → running       (resume)
//!   running   → completed     (finish)
//!   running   → failed        (crash)
//!   queued    → cancelled     (drop from queue)
//!   running   → cancelled     (stop button)
//!
//! Only the transitions reachable from a button press are exposed
//! via HTTP — internal transitions (running → completed) come from
//! the runner via the gRPC layer (P2.8).

use crate::db::{self, NewRun, RunStatus};
use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Deserialize, utoipa::IntoParams)]
pub struct ListQuery {
    /// Optional status filter — one of: queued, running, paused, completed, failed, cancelled.
    pub status: Option<String>,
    /// Page size. Capped at 1000 server-side.
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct ListResponse {
    pub items: Vec<db::RunSummary>,
    pub total: i64,
}

#[utoipa::path(
    get, path = "/runs",
    params(ListQuery),
    responses((status = 200, body = ListResponse))
)]
pub async fn list(
    State(s): State<AppState>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<ListResponse>> {
    let status = q.status.as_deref().map(RunStatus::from_str).transpose()?;
    let limit = q.limit.unwrap_or(100).clamp(1, 1000);
    let offset = q.offset.unwrap_or(0).max(0);

    let items = db::list_runs(&s.db, status, limit, offset).await?;
    let total = db::count_runs(&s.db, status).await?;
    Ok(Json(ListResponse { items, total }))
}

#[utoipa::path(get, path = "/runs/{id}", responses((status = 200, body = db::Run)))]
pub async fn get_one(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<db::Run>> {
    let r = db::get_run(&s.db, &id).await?;
    Ok(Json(r))
}

#[derive(Deserialize, utoipa::ToSchema)]
pub struct CreateBody {
    pub title: Option<String>,
    /// Free-form TrainCfg blob — the runner is the authority on its
    /// schema; the server just persists it.
    pub cfg: serde_json::Value,
    pub sweep_id: Option<String>,
}

#[utoipa::path(
    post, path = "/runs",
    request_body = CreateBody,
    responses((status = 201, body = db::Run))
)]
pub async fn create(
    State(s): State<AppState>,
    Json(body): Json<CreateBody>,
) -> ApiResult<(StatusCode, Json<db::Run>)> {
    if !body.cfg.is_object() {
        return Err(ApiError::InvalidArg("cfg must be a JSON object".into()));
    }
    let run = db::insert_run(
        &s.db,
        NewRun {
            title: body.title,
            cfg_json: body.cfg,
            sweep_id: body.sweep_id,
        },
    )
    .await?;

    s.hub.publish(
        "runs.changed",
        "runs.changed",
        &serde_json::json!({"id": &run.id, "status": run.status}),
    );
    Ok((StatusCode::CREATED, Json(run)))
}

/// Apply one of the lifecycle triggers. Each is a thin wrapper that
/// validates the requested transition before persisting.
async fn trigger(
    s: AppState,
    id: String,
    next: RunStatus,
    allowed_from: &[RunStatus],
) -> ApiResult<Json<db::Run>> {
    let current = db::get_run(&s.db, &id).await?;
    if !allowed_from.contains(&current.status) {
        return Err(ApiError::Conflict(format!(
            "cannot transition {:?} → {:?}",
            current.status, next
        )));
    }
    let updated = db::update_run_status(&s.db, &id, next).await?;
    s.hub.publish(
        "runs.changed",
        "runs.changed",
        &serde_json::json!({"id": &updated.id, "status": updated.status}),
    );
    Ok(Json(updated))
}

#[utoipa::path(post, path = "/runs/{id}/start", responses((status = 200, body = db::Run)))]
pub async fn start(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<db::Run>> {
    trigger(
        s,
        id,
        RunStatus::Running,
        &[RunStatus::Queued, RunStatus::Paused],
    )
    .await
}

#[utoipa::path(post, path = "/runs/{id}/pause", responses((status = 200, body = db::Run)))]
pub async fn pause(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<db::Run>> {
    trigger(s, id, RunStatus::Paused, &[RunStatus::Running]).await
}

#[utoipa::path(post, path = "/runs/{id}/resume", responses((status = 200, body = db::Run)))]
pub async fn resume(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<db::Run>> {
    trigger(s, id, RunStatus::Running, &[RunStatus::Paused]).await
}

#[utoipa::path(post, path = "/runs/{id}/stop", responses((status = 200, body = db::Run)))]
pub async fn stop(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<db::Run>> {
    trigger(
        s,
        id,
        RunStatus::Cancelled,
        &[RunStatus::Queued, RunStatus::Running, RunStatus::Paused],
    )
    .await
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/runs", get(list).post(create))
        .route("/runs/:id", get(get_one))
        .route("/runs/:id/start", post(start))
        .route("/runs/:id/pause", post(pause))
        .route("/runs/:id/resume", post(resume))
        .route("/runs/:id/stop", post(stop))
}
