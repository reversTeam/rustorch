//! `GET /activity` — recent events feed shown on the Dashboard.

use crate::db;
use crate::error::ApiResult;
use crate::state::AppState;
use axum::{
    extract::{Query, State},
    routing::get,
    Json, Router,
};
use serde::Deserialize;

#[derive(Deserialize, utoipa::IntoParams)]
pub struct ActivityQuery {
    /// How many recent events to return. Capped server-side at 200.
    pub limit: Option<i64>,
}

#[utoipa::path(
    get, path = "/activity",
    params(ActivityQuery),
    responses((status = 200, body = [db::ActivityEvent]))
)]
pub async fn list(
    State(s): State<AppState>,
    Query(q): Query<ActivityQuery>,
) -> ApiResult<Json<Vec<db::ActivityEvent>>> {
    let limit = q.limit.unwrap_or(50).clamp(1, 200);
    let items = db::list_activity(&s.db, limit).await?;
    Ok(Json(items))
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/activity", get(list))
}
