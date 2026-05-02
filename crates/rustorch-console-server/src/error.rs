//! Stable error contract — frontend & SDKs match on `code`, never on
//! the human `message`. The list is fixed by the v0.7.2 doc; new
//! variants must be added with care because the frontend has UI
//! treatment per code.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

/// One of the stable error codes from doc v0.7.2 §"Reference/Console
/// API/Error contract". Tweak the variants only by adding new ones;
/// renaming silently breaks every UI badge.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    NotFound,
    InvalidArg,
    Conflict,
    Unauthorized,
    QuotaExceeded,
    Internal,
}

impl ErrorCode {
    /// HTTP status that ships with each code. The mapping is
    /// deliberately boring (we don't try to be clever about 422 vs
    /// 400) — clients should branch on `code`, not status.
    pub fn http_status(self) -> StatusCode {
        match self {
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::InvalidArg => StatusCode::BAD_REQUEST,
            Self::Conflict => StatusCode::CONFLICT,
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::QuotaExceeded => StatusCode::TOO_MANY_REQUESTS,
            Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// JSON body emitted on every non-2xx response. Matches doc v0.7.2
/// shape `{ code, message, details? }`.
#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub code: ErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

/// Crate-wide error type. Implements `IntoResponse` so handlers can
/// use `?` directly and let axum format the JSON body.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid argument: {0}")]
    InvalidArg(String),

    #[error("conflict: {0}")]
    Conflict(String),

    #[error("unauthorized")]
    Unauthorized,

    #[error("quota exceeded: {0}")]
    QuotaExceeded(String),

    /// Anything we don't want to expose to the caller. The original
    /// error is logged at `error!` level by the IntoResponse impl.
    #[error("internal: {0}")]
    Internal(String),

    /// Convenience pass-through so `?` works on sqlx errors.
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

impl ApiError {
    fn parts(self) -> (ErrorCode, String, Option<serde_json::Value>) {
        match self {
            Self::NotFound(m) => (ErrorCode::NotFound, m, None),
            Self::InvalidArg(m) => (ErrorCode::InvalidArg, m, None),
            Self::Conflict(m) => (ErrorCode::Conflict, m, None),
            Self::Unauthorized => (ErrorCode::Unauthorized, "unauthorized".into(), None),
            Self::QuotaExceeded(m) => (ErrorCode::QuotaExceeded, m, None),
            Self::Internal(m) => {
                tracing::error!(message = %m, "internal error surfaced");
                (ErrorCode::Internal, "internal server error".into(), None)
            },
            Self::Db(e) => {
                tracing::error!(error = %e, "sqlx error");
                (ErrorCode::Internal, "database error".into(), None)
            },
            Self::Json(e) => {
                tracing::error!(error = %e, "json error");
                (ErrorCode::InvalidArg, e.to_string(), None)
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (code, message, details) = self.parts();
        let body = ErrorBody {
            code,
            message,
            details,
        };
        (code.http_status(), Json(body)).into_response()
    }
}

/// Sugar for handlers that return `Result<T, ApiError>`.
pub type ApiResult<T> = Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn http_status_mapping_is_stable() {
        assert_eq!(ErrorCode::NotFound.http_status(), StatusCode::NOT_FOUND);
        assert_eq!(ErrorCode::InvalidArg.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(ErrorCode::Conflict.http_status(), StatusCode::CONFLICT);
        assert_eq!(
            ErrorCode::Unauthorized.http_status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            ErrorCode::QuotaExceeded.http_status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            ErrorCode::Internal.http_status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn error_code_serializes_screaming_snake() {
        let s = serde_json::to_string(&ErrorCode::NotFound).unwrap();
        assert_eq!(s, "\"NOT_FOUND\"");
        let s = serde_json::to_string(&ErrorCode::QuotaExceeded).unwrap();
        assert_eq!(s, "\"QUOTA_EXCEEDED\"");
    }

    #[test]
    fn body_skips_details_when_none() {
        let body = ErrorBody {
            code: ErrorCode::NotFound,
            message: "run".into(),
            details: None,
        };
        let s = serde_json::to_string(&body).unwrap();
        assert!(!s.contains("details"));
    }
}
