//! Trivial identity / workspace endpoints — placeholders the UI
//! polls on every page load. Real multi-tenant support is out of
//! scope for v0.7.2; today every authenticated request maps to a
//! single shared workspace.

use crate::error::ApiResult;
use axum::{routing::get, Json, Router};
use serde::Serialize;

#[derive(Serialize, utoipa::ToSchema)]
pub struct MeResponse {
    pub user: String,
    pub workspace: String,
    pub roles: Vec<String>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct WorkspaceResponse {
    pub name: String,
    pub description: String,
    /// Server build version, bubbled through to the UI footer.
    pub server_version: &'static str,
}

#[utoipa::path(get, path = "/me", responses((status = 200, body = MeResponse)))]
pub async fn me() -> ApiResult<Json<MeResponse>> {
    Ok(Json(MeResponse {
        // Bearer-token mode is single-user by design — the ID is
        // a stable label rather than an authenticated principal.
        user: "rustorch-local".into(),
        workspace: "default".into(),
        roles: vec!["admin".into()],
    }))
}

#[utoipa::path(get, path = "/workspace", responses((status = 200, body = WorkspaceResponse)))]
pub async fn workspace() -> ApiResult<Json<WorkspaceResponse>> {
    Ok(Json(WorkspaceResponse {
        name: "default".into(),
        description: "rustorch local workspace".into(),
        server_version: env!("CARGO_PKG_VERSION"),
    }))
}

pub fn routes<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/me", get(me))
        .route("/workspace", get(workspace))
}
