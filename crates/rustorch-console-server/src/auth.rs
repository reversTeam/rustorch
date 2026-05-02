//! Bearer-token middleware. Default policy:
//!   * if `RUSTORCH_TOKEN` is unset → server is open (dev mode);
//!   * if it is set → every request must carry a matching
//!     `Authorization: Bearer <token>` header, *except* the explicit
//!     allow-list (health, openapi, swagger UI).
//!
//! The middleware does not implement scopes / per-route policies —
//! the v0.7.2 spec uses a single shared workspace token. A finer
//! policy can be layered on top later without breaking the existing
//! contract.

use axum::{
    extract::Request,
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// Routes that must always be reachable without auth so a load
/// balancer can probe and the doc viewer can fetch the spec.
const ALWAYS_ALLOW: &[&str] = &["/health", "/openapi.json", "/swagger-ui"];

/// Token configuration extracted once at startup. Cloned per request
/// (Arc<str> would be marginally better; we keep `Option<String>`
/// because the value is set at most once).
#[derive(Clone, Debug, Default)]
pub struct AuthConfig {
    /// `None` → open mode. `Some(_)` → require Bearer match.
    pub token: Option<String>,
}

impl AuthConfig {
    /// Read `RUSTORCH_TOKEN` from the env. Empty string is treated
    /// as unset — that's the typical CI setup.
    pub fn from_env() -> Self {
        Self {
            token: std::env::var("RUSTORCH_TOKEN")
                .ok()
                .filter(|s| !s.is_empty()),
        }
    }
}

/// Axum middleware. Mounted via `Router::layer(middleware::from_fn_with_state(...))`.
pub async fn require_bearer(
    axum::extract::State(cfg): axum::extract::State<AuthConfig>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path();
    if cfg.token.is_none()
        || ALWAYS_ALLOW
            .iter()
            .any(|p| path == *p || path.starts_with(p))
    {
        return next.run(req).await;
    }
    let expected = cfg.token.as_deref().unwrap();

    let header_val = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match header_val {
        Some(v) if v.strip_prefix("Bearer ") == Some(expected) => next.run(req).await,
        _ => unauthorized().into_response(),
    }
}

fn unauthorized() -> impl IntoResponse {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "code": "UNAUTHORIZED",
            "message": "missing or invalid Bearer token"
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_defaults_to_open() {
        // Snapshot env so parallel tests don't poison each other.
        let prev = std::env::var("RUSTORCH_TOKEN").ok();
        unsafe { std::env::remove_var("RUSTORCH_TOKEN") };
        let c = AuthConfig::from_env();
        assert!(c.token.is_none());
        if let Some(p) = prev {
            unsafe { std::env::set_var("RUSTORCH_TOKEN", p) };
        }
    }

    #[test]
    fn allow_list_covers_health_and_openapi() {
        for path in ["/health", "/openapi.json", "/swagger-ui/index.html"] {
            assert!(
                ALWAYS_ALLOW
                    .iter()
                    .any(|p| path == *p || path.starts_with(p)),
                "{path} should be allow-listed"
            );
        }
    }
}
