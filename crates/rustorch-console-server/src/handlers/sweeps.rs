//! `/sweeps` — create + inspect hyperparameter sweeps. Persists in
//! the `sweeps` table introduced by migration 0003. Trial planning
//! is delegated to `rustorch-sweep`.
//!
//! v1 surface:
//! * `POST /sweeps`            — register + persist + return trials.
//! * `GET  /sweeps`            — list sweeps, recent first.
//! * `GET  /sweeps/:id`        — fetch a sweep + its planned trials.
//! * `POST /sweeps/:id/cancel` — mark cancelled (the run-queue uses
//!   the field to skip queued trials).

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use rustorch_sweep::{
    AshaConfig, BayesConfig, RandomConfig, Strategy, SweepBuilder, SweepSpec, Trial,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, utoipa::ToSchema)]
pub struct CreateSweepBody {
    /// Lower-case strategy name: `grid` | `random` | `asha` | `bayes`.
    pub strategy: String,
    /// Strategy-specific knobs. Optional for grid.
    #[serde(default)]
    pub config: serde_json::Value,
    /// Axes — array of `{name, values: [...]}`. Order matters for
    /// reproducibility.
    pub axes: Vec<AxisInput>,
    /// Base TrainCfg the trials get merged into.
    pub base_cfg: serde_json::Value,
}

#[derive(Deserialize, utoipa::ToSchema)]
pub struct AxisInput {
    pub name: String,
    pub values: serde_json::Value,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct SweepResponse {
    pub id: String,
    pub strategy: String,
    pub status: String,
    pub trials: Vec<Trial>,
    pub created_at: chrono::DateTime<Utc>,
}

#[utoipa::path(
    post, path = "/sweeps",
    request_body = CreateSweepBody,
    responses((status = 201, body = SweepResponse))
)]
pub async fn create(
    State(s): State<AppState>,
    Json(body): Json<CreateSweepBody>,
) -> ApiResult<(StatusCode, Json<SweepResponse>)> {
    let strategy_name = body.strategy.to_ascii_lowercase();
    let mut builder: SweepBuilder = match strategy_name.as_str() {
        "grid" => SweepSpec::grid(),
        "random" => {
            let cfg: RandomConfig = serde_json::from_value(body.config.clone())
                .map_err(|e| ApiError::InvalidArg(format!("random config: {e}")))?;
            SweepSpec::random(cfg.trials, cfg.seed)
        },
        "asha" => {
            let cfg: AshaConfig = serde_json::from_value(body.config.clone())
                .map_err(|e| ApiError::InvalidArg(format!("asha config: {e}")))?;
            SweepSpec::asha(cfg)
        },
        "bayes" => {
            let cfg: BayesConfig = serde_json::from_value(body.config.clone())
                .map_err(|e| ApiError::InvalidArg(format!("bayes config: {e}")))?;
            SweepSpec::bayes(cfg)
        },
        other => return Err(ApiError::InvalidArg(format!("unknown strategy: {other}"))),
    };
    for axis in &body.axes {
        builder = builder
            .add(&axis.name, &axis.values)
            .map_err(|e| ApiError::InvalidArg(format!("axis {}: {e}", axis.name)))?;
    }
    let spec = builder.build();
    let trials = spec
        .plan(&body.base_cfg)
        .map_err(|e| ApiError::InvalidArg(format!("plan: {e}")))?;

    let id = ulid::Ulid::new().to_string();
    let now = Utc::now();

    // Persist the spec + axes so a future GET returns the same shape.
    let spec_json = serde_json::to_string(&serde_json::json!({
        "strategy": strategy_name,
        "config": body.config,
        "axes": body.axes.iter().map(|a| serde_json::json!({"name": a.name, "values": a.values})).collect::<Vec<_>>(),
    }))?;
    let base = serde_json::to_string(&body.base_cfg)?;

    sqlx::query(
        "INSERT INTO sweeps (id, strategy, status, spec_json, base_cfg_json, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(&strategy_name)
    .bind("pending")
    .bind(&spec_json)
    .bind(&base)
    .bind(now.to_rfc3339())
    .bind(now.to_rfc3339())
    .execute(&s.db)
    .await?;

    Ok((
        StatusCode::CREATED,
        Json(SweepResponse {
            id,
            strategy: strategy_name,
            status: "pending".into(),
            trials,
            created_at: now,
        }),
    ))
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct SweepSummary {
    pub id: String,
    pub strategy: String,
    pub status: String,
    pub created_at: chrono::DateTime<Utc>,
}

#[utoipa::path(get, path = "/sweeps", responses((status = 200, body = [SweepSummary])))]
pub async fn list(State(s): State<AppState>) -> ApiResult<Json<Vec<SweepSummary>>> {
    let rows: Vec<(String, String, String, String)> = sqlx::query_as(
        "SELECT id, strategy, status, created_at FROM sweeps
         ORDER BY created_at DESC LIMIT 200",
    )
    .fetch_all(&s.db)
    .await?;
    let items = rows
        .into_iter()
        .map(|(id, strategy, status, ts)| SweepSummary {
            id,
            strategy,
            status,
            created_at: chrono::DateTime::parse_from_rfc3339(&ts)
                .map(|d| d.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
        .collect();
    Ok(Json(items))
}

#[utoipa::path(get, path = "/sweeps/{id}", responses((status = 200, body = SweepResponse)))]
pub async fn get_one(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<SweepResponse>> {
    let row: Option<(String, String, String, String, String)> = sqlx::query_as(
        "SELECT strategy, status, spec_json, base_cfg_json, created_at
         FROM sweeps WHERE id = ?",
    )
    .bind(&id)
    .fetch_optional(&s.db)
    .await?;
    let (strategy, status, spec_json, base_cfg_json, ts) =
        row.ok_or_else(|| ApiError::NotFound(format!("sweep {id}")))?;

    // Replan from the persisted spec so the response always carries
    // the trial list, even after a Console restart.
    let parsed: serde_json::Value = serde_json::from_str(&spec_json)?;
    let body = CreateSweepBody {
        strategy: strategy.clone(),
        config: parsed
            .get("config")
            .cloned()
            .unwrap_or(serde_json::json!({})),
        axes: parsed
            .get("axes")
            .and_then(|v| serde_json::from_value::<Vec<AxisInput>>(v.clone()).ok())
            .unwrap_or_default(),
        base_cfg: serde_json::from_str(&base_cfg_json)?,
    };
    // Borrow `replan_from` so we don't duplicate the strategy match.
    let mut builder: SweepBuilder = match strategy.as_str() {
        "grid" => SweepSpec::grid(),
        "random" => {
            let cfg: RandomConfig =
                serde_json::from_value(body.config.clone()).unwrap_or(RandomConfig {
                    trials: 1,
                    seed: None,
                });
            SweepSpec::random(cfg.trials, cfg.seed)
        },
        "asha" => {
            let cfg: AshaConfig =
                serde_json::from_value(body.config.clone()).unwrap_or(AshaConfig {
                    eta: 3,
                    brackets: 1,
                    min_resource: 1,
                    max_resource: 27,
                });
            SweepSpec::asha(cfg)
        },
        "bayes" => {
            let cfg: BayesConfig =
                serde_json::from_value(body.config.clone()).unwrap_or(BayesConfig {
                    init_trials: 5,
                    trials: 30,
                    acquisition: "ei".into(),
                    seed: None,
                });
            SweepSpec::bayes(cfg)
        },
        _ => return Err(ApiError::Internal("bad persisted strategy".into())),
    };
    for axis in &body.axes {
        builder = builder
            .add(&axis.name, &axis.values)
            .map_err(|e| ApiError::Internal(format!("replan axis {}: {e}", axis.name)))?;
    }
    let trials = builder
        .build()
        .plan(&body.base_cfg)
        .map_err(|e| ApiError::Internal(format!("replan: {e}")))?;
    let _ = Strategy::Grid; // touch the type for utoipa

    Ok(Json(SweepResponse {
        id,
        strategy,
        status,
        trials,
        created_at: chrono::DateTime::parse_from_rfc3339(&ts)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
    }))
}

#[utoipa::path(post, path = "/sweeps/{id}/cancel", responses((status = 200)))]
pub async fn cancel(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<SweepSummary>> {
    let now = Utc::now();
    let res = sqlx::query(
        "UPDATE sweeps SET status = 'cancelled', updated_at = ? WHERE id = ? AND status != 'completed'",
    )
    .bind(now.to_rfc3339())
    .bind(&id)
    .execute(&s.db)
    .await?;
    if res.rows_affected() == 0 {
        return Err(ApiError::NotFound(format!("sweep {id}")));
    }
    let row: (String, String) =
        sqlx::query_as("SELECT strategy, created_at FROM sweeps WHERE id = ?")
            .bind(&id)
            .fetch_one(&s.db)
            .await?;
    Ok(Json(SweepSummary {
        id,
        strategy: row.0,
        status: "cancelled".into(),
        created_at: chrono::DateTime::parse_from_rfc3339(&row.1)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
    }))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/sweeps", get(list).post(create))
        .route("/sweeps/:id", get(get_one))
        .route("/sweeps/:id/cancel", post(cancel))
}
