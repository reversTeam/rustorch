//! `/deploy` + `/datasets POST` — push artifacts somewhere.
//!
//! v1 records the deploy intent in the `activity` table and returns
//! the manifest. The actual gRPC call to the serving runtime lands
//! in P2.6; the dataset registration to a real registry lands later
//! in this same plan slice.

use crate::db;
use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize, utoipa::ToSchema)]
pub struct DeployBody {
    pub run_id: String,
    pub checkpoint: String,
    /// HTTP / gRPC URL of the target serving cluster.
    pub target_url: String,
    #[serde(default)]
    pub autoscale: serde_json::Value,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct DeployResponse {
    pub run_id: String,
    pub checkpoint: String,
    pub target_url: String,
    pub status: &'static str,
}

#[utoipa::path(
    post, path = "/deploy",
    request_body = DeployBody,
    responses((status = 202, body = DeployResponse))
)]
pub async fn deploy(
    State(s): State<AppState>,
    Json(body): Json<DeployBody>,
) -> ApiResult<(StatusCode, Json<DeployResponse>)> {
    if !body.target_url.starts_with("http://")
        && !body.target_url.starts_with("https://")
        && !body.target_url.starts_with("grpc://")
    {
        return Err(ApiError::InvalidArg(format!(
            "target_url must start with http(s):// or grpc:// (got {})",
            body.target_url
        )));
    }
    // 404 early if the run doesn't exist.
    let _ = db::get_run(&s.db, &body.run_id).await?;

    db::insert_activity(
        &s.db,
        "deploy_requested",
        Some(&body.run_id),
        json!({
            "checkpoint": body.checkpoint,
            "target_url": body.target_url,
            "autoscale": body.autoscale,
        }),
    )
    .await?;

    Ok((
        StatusCode::ACCEPTED,
        Json(DeployResponse {
            run_id: body.run_id,
            checkpoint: body.checkpoint,
            target_url: body.target_url,
            status: "queued",
        }),
    ))
}

#[derive(Deserialize, utoipa::ToSchema)]
pub struct RegisterDatasetBody {
    pub name: String,
    pub url: String,
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct RegisterDatasetResponse {
    pub name: String,
    pub url: String,
    pub status: &'static str,
}

#[utoipa::path(
    post, path = "/datasets",
    request_body = RegisterDatasetBody,
    responses((status = 201, body = RegisterDatasetResponse))
)]
pub async fn register_dataset(
    State(s): State<AppState>,
    Json(body): Json<RegisterDatasetBody>,
) -> ApiResult<(StatusCode, Json<RegisterDatasetResponse>)> {
    if body.name.is_empty() {
        return Err(ApiError::InvalidArg("name required".into()));
    }
    if body.url.is_empty() {
        return Err(ApiError::InvalidArg("url required".into()));
    }
    db::insert_activity(
        &s.db,
        "dataset_registered",
        None,
        json!({
            "name": body.name,
            "url": body.url,
            "task": body.task,
            "sha256": body.sha256,
        }),
    )
    .await?;
    Ok((
        StatusCode::CREATED,
        Json(RegisterDatasetResponse {
            name: body.name,
            url: body.url,
            status: "registered",
        }),
    ))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/deploy", post(deploy))
        .route("/datasets", post(register_dataset))
}
