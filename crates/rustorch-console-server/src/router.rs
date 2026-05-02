//! Top-level `Router` builder. Used by `main.rs` (production) and
//! by `tests/api.rs` (integration). Keeping this assembly in lib
//! code lets tests spawn a real axum server on `127.0.0.1:0` without
//! depending on a binary build.

use crate::auth::{require_bearer, AuthConfig};
use crate::handlers::{cluster, me, runs, sse_routes};
use crate::openapi::ApiDoc;
use crate::state::AppState;
use axum::{middleware, routing::get, Json, Router};
use serde_json::json;
use tower_http::cors::CorsLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

/// Build the full app router. `auth` is taken explicitly (not from
/// the env) so tests can build a router with `AuthConfig::default()`
/// regardless of the environment they're running in.
pub fn build(state: AppState, auth: AuthConfig) -> Router {
    // Routes that DO carry app state.
    let with_state = Router::new()
        .merge(me::routes())
        .merge(cluster::routes())
        .merge(runs::routes())
        .merge(sse_routes::routes())
        .with_state(state);

    // Routes that don't (health, openapi). Kept separate so they
    // never depend on the DB being healthy. SwaggerUi mounts both
    // `/swagger-ui/*` AND `/openapi.json` for us.
    let public = Router::new()
        .route("/health", get(health))
        .merge(SwaggerUi::new("/swagger-ui").url("/openapi.json", ApiDoc::openapi()));

    Router::new()
        .merge(with_state)
        .merge(public)
        .layer(middleware::from_fn_with_state(auth, require_bearer))
        .layer(CorsLayer::permissive())
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}
