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

#[derive(Deserialize, utoipa::ToSchema, Default)]
pub struct ForkBody {
    /// Optional new title — defaults to "fork of <source title>".
    pub title: Option<String>,
    /// Hyperparameter overrides merged into the source `cfg` (shallow
    /// merge — top-level keys win on the override side).
    #[serde(default)]
    pub overrides: serde_json::Value,
    /// Optional checkpoint path the new runner should resume from.
    pub resume_from: Option<String>,
}

/// `POST /runs/:id/fork` — create a new queued run derived from
/// `:id`. The new run inherits the source `cfg`, optionally
/// overlaid with `body.overrides`, and remembers its parent via
/// `cfg.parent_run_id` so the UI can render the lineage.
///
/// The actual training process spawn happens through the runner /
/// CLI (P2.8). At the API layer fork is purely a data-layer
/// operation: clone + tag.
#[utoipa::path(
    post, path = "/runs/{id}/fork",
    request_body = ForkBody,
    responses((status = 201, body = db::Run))
)]
pub async fn fork(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ForkBody>,
) -> ApiResult<(StatusCode, Json<db::Run>)> {
    let parent = db::get_run(&s.db, &id).await?;
    let mut cfg: serde_json::Value = serde_json::from_str(&parent.cfg_json).unwrap_or_default();

    // Shallow merge top-level keys from `overrides` into `cfg`.
    if let (Some(cfg_obj), Some(over_obj)) = (cfg.as_object_mut(), body.overrides.as_object()) {
        for (k, v) in over_obj {
            cfg_obj.insert(k.clone(), v.clone());
        }
    }

    // Lineage tags so the frontend can show "forked from X".
    if let Some(obj) = cfg.as_object_mut() {
        obj.insert(
            "parent_run_id".into(),
            serde_json::Value::String(parent.id.clone()),
        );
        if let Some(rf) = body.resume_from.as_ref() {
            obj.insert("resume_from".into(), serde_json::Value::String(rf.clone()));
        }
    }

    let title = body
        .title
        .or_else(|| parent.title.as_deref().map(|t| format!("fork of {t}")));

    let run = db::insert_run(
        &s.db,
        NewRun {
            title,
            cfg_json: cfg,
            sweep_id: parent.sweep_id.clone(),
        },
    )
    .await?;

    s.hub.publish(
        "runs.changed",
        "runs.changed",
        &serde_json::json!({"id": &run.id, "status": run.status, "forked_from": parent.id}),
    );
    Ok((StatusCode::CREATED, Json(run)))
}

// ---- read-only views (curves / hparams / code / log / checkpoints / artifacts / system) -----

#[derive(Serialize, utoipa::ToSchema)]
pub struct CurvesResponse {
    /// Distinct metric names available for this run (`loss`, `val_acc`, `lr`, …).
    pub names: Vec<String>,
    pub points: Vec<db::MetricPoint>,
}

#[utoipa::path(get, path = "/runs/{id}/curves", responses((status = 200, body = CurvesResponse)))]
pub async fn curves(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<CurvesResponse>> {
    // 404 early if the run is missing.
    let _ = db::get_run(&s.db, &id).await?;
    let points = db::list_metrics(&s.db, &id).await?;
    let mut names: Vec<String> = points.iter().map(|p| p.name.clone()).collect();
    names.sort();
    names.dedup();
    Ok(Json(CurvesResponse { names, points }))
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct HparamsResponse {
    pub run_id: String,
    pub cfg: serde_json::Value,
}

#[utoipa::path(get, path = "/runs/{id}/hparams", responses((status = 200, body = HparamsResponse)))]
pub async fn hparams(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<HparamsResponse>> {
    let run = db::get_run(&s.db, &id).await?;
    let cfg: serde_json::Value = serde_json::from_str(&run.cfg_json).unwrap_or_default();
    Ok(Json(HparamsResponse {
        run_id: run.id,
        cfg,
    }))
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct CodeResponse {
    pub run_id: String,
    /// Source `.rs` content of the training entry point. Until the
    /// runner uploads it (P2.8), this is a placeholder pulled from
    /// `cfg.code` if present, otherwise an empty string.
    pub source: String,
    /// Optional commit SHA the runner was at when the run started.
    pub commit_sha: Option<String>,
}

#[utoipa::path(get, path = "/runs/{id}/code", responses((status = 200, body = CodeResponse)))]
pub async fn code(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<CodeResponse>> {
    let run = db::get_run(&s.db, &id).await?;
    let cfg: serde_json::Value = serde_json::from_str(&run.cfg_json).unwrap_or_default();
    let source = cfg
        .get("code")
        .and_then(|v| v.as_str())
        .unwrap_or("// no source recorded yet (runner upload lands in P2.8)\n")
        .to_string();
    let commit_sha = cfg
        .get("commit_sha")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Ok(Json(CodeResponse {
        run_id: run.id,
        source,
        commit_sha,
    }))
}

#[derive(Deserialize, utoipa::IntoParams)]
pub struct LogQuery {
    /// How many last lines to return. Capped at 10_000.
    pub tail: Option<i64>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct LogLine {
    pub ts: chrono::DateTime<chrono::Utc>,
    pub level: &'static str,
    pub msg: String,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct LogResponse {
    pub run_id: String,
    pub lines: Vec<LogLine>,
}

#[utoipa::path(
    get, path = "/runs/{id}/log",
    params(LogQuery),
    responses((status = 200, body = LogResponse))
)]
pub async fn log(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Query(_q): Query<LogQuery>,
) -> ApiResult<Json<LogResponse>> {
    // The actual log table lands with the gRPC ingest layer (P2.8 /
    // P2.5). For now we synthesize a minimal placeholder so the UI
    // can render something deterministic during development.
    let run = db::get_run(&s.db, &id).await?;
    let lines = vec![LogLine {
        ts: run.created_at,
        level: "info",
        msg: format!("run {} created (status: {:?})", run.id, run.status),
    }];
    Ok(Json(LogResponse {
        run_id: run.id,
        lines,
    }))
}

#[utoipa::path(
    get, path = "/runs/{id}/checkpoints",
    responses((status = 200, body = [db::Checkpoint]))
)]
pub async fn checkpoints(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Vec<db::Checkpoint>>> {
    let _ = db::get_run(&s.db, &id).await?;
    let cps = db::list_checkpoints(&s.db, &id).await?;
    Ok(Json(cps))
}

#[derive(Deserialize, utoipa::ToSchema)]
pub struct NewCheckpointBody {
    pub path: String,
    pub step: i64,
    /// Free-form metrics snapshot (val_acc, loss, …).
    pub metrics: serde_json::Value,
}

/// `POST /runs/:id/checkpoints` — register a saved checkpoint.
/// Today this is exposed for tests + an HTTP fallback when a runner
/// can't speak gRPC (P2.8). Publishes `run.{id}.checkpoint` so the
/// Run detail UI can refresh without polling.
#[utoipa::path(
    post, path = "/runs/{id}/checkpoints",
    request_body = NewCheckpointBody,
    responses((status = 201, body = db::Checkpoint))
)]
pub async fn save_checkpoint(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<NewCheckpointBody>,
) -> ApiResult<(StatusCode, Json<db::Checkpoint>)> {
    let cp = db::insert_checkpoint(
        &s.db,
        db::NewCheckpoint {
            run_id: id.clone(),
            path: body.path,
            step: body.step,
            metrics: body.metrics,
        },
    )
    .await?;

    let topic = format!("run.{id}.checkpoint");
    s.hub.publish(
        &topic,
        "checkpoint",
        &serde_json::json!({
            "id": cp.id,
            "run_id": cp.run_id,
            "path": cp.path,
            "step": cp.step,
            "metrics": cp.metrics,
        }),
    );
    Ok((StatusCode::CREATED, Json(cp)))
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct ArtifactEntry {
    pub kind: &'static str,
    pub path: String,
    pub bytes: u64,
}

#[utoipa::path(
    get, path = "/runs/{id}/artifacts",
    responses((status = 200, body = [ArtifactEntry]))
)]
pub async fn artifacts(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Vec<ArtifactEntry>>> {
    // Artifacts arrive via the runner gRPC stream (P2.8). Until then
    // we materialize the run's checkpoints as `kind: "checkpoint"`
    // artifacts so the UI has something to render.
    let _ = db::get_run(&s.db, &id).await?;
    let cps = db::list_checkpoints(&s.db, &id).await?;
    let items: Vec<ArtifactEntry> = cps
        .into_iter()
        .map(|c| ArtifactEntry {
            kind: "checkpoint",
            path: c.path,
            bytes: 0,
        })
        .collect();
    Ok(Json(items))
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct SystemSample {
    pub gpu_id: u32,
    pub util: u32,
    pub vram_used_mb: u32,
    pub temp_c: u32,
    pub power_w: u32,
    pub throughput_samples_per_s: f64,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct SystemResponse {
    pub run_id: String,
    pub samples: Vec<SystemSample>,
}

#[utoipa::path(get, path = "/runs/{id}/system", responses((status = 200, body = SystemResponse)))]
pub async fn system(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<SystemResponse>> {
    // Mock samples — real per-GPU / throughput data flows through
    // the gRPC `GpuTelemetry` event in P2.8.
    let run = db::get_run(&s.db, &id).await?;
    Ok(Json(SystemResponse {
        run_id: run.id,
        samples: vec![SystemSample {
            gpu_id: 0,
            util: 0,
            vram_used_mb: 256,
            temp_c: 38,
            power_w: 60,
            throughput_samples_per_s: 0.0,
        }],
    }))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/runs", get(list).post(create))
        .route("/runs/:id", get(get_one))
        .route("/runs/:id/start", post(start))
        .route("/runs/:id/pause", post(pause))
        .route("/runs/:id/resume", post(resume))
        .route("/runs/:id/stop", post(stop))
        .route("/runs/:id/fork", post(fork))
        .route("/runs/:id/curves", get(curves))
        .route("/runs/:id/hparams", get(hparams))
        .route("/runs/:id/code", get(code))
        .route("/runs/:id/log", get(log))
        .route(
            "/runs/:id/checkpoints",
            get(checkpoints).post(save_checkpoint),
        )
        .route("/runs/:id/artifacts", get(artifacts))
        .route("/runs/:id/system", get(system))
}
