//! HTTP server façade. Exposes `POST /predict`, `GET /health`,
//! `GET /metrics` (Prometheus-style stub).

use crate::batcher::{BatchConfig, Batcher, BatcherError, InferFn};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error("listen: {0}")]
    Listen(#[from] std::io::Error),
    #[error("serve: {0}")]
    Serve(String),
}

#[derive(Deserialize)]
struct PredictBody {
    inputs: serde_json::Value,
}

#[derive(Serialize)]
struct PredictResponse {
    outputs: serde_json::Value,
}

#[derive(Clone)]
struct AppState {
    batcher: Batcher,
    metrics: Arc<Metrics>,
}

#[derive(Default)]
struct Metrics {
    requests_total: parking_lot::Mutex<u64>,
    overload_total: parking_lot::Mutex<u64>,
}

async fn health() -> &'static str {
    "ok"
}

async fn metrics(State(s): State<AppState>) -> String {
    // Prometheus exposition format. Tiny v1 — the real exporter
    // tracks p50/p99 histograms via `prometheus` crate.
    format!(
        "# HELP rustorch_serve_requests_total Total /predict requests.\n\
         # TYPE rustorch_serve_requests_total counter\n\
         rustorch_serve_requests_total {}\n\
         # HELP rustorch_serve_overload_total Total 503 responses.\n\
         # TYPE rustorch_serve_overload_total counter\n\
         rustorch_serve_overload_total {}\n",
        s.metrics.requests_total.lock(),
        s.metrics.overload_total.lock(),
    )
}

async fn predict(State(s): State<AppState>, Json(body): Json<PredictBody>) -> impl IntoResponse {
    *s.metrics.requests_total.lock() += 1;
    match s.batcher.predict(body.inputs).await {
        Ok(out) => Json(PredictResponse { outputs: out }).into_response(),
        Err(BatcherError::Overloaded) => {
            *s.metrics.overload_total.lock() += 1;
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "overloaded"})),
            )
                .into_response()
        },
        Err(BatcherError::Shutdown) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "shutdown"})),
        )
            .into_response(),
    }
}

/// Fluent builder. v1 supports HTTP only; gRPC support arrives via
/// `listen_grpc` in a follow-up commit.
pub struct ServeBuilder {
    infer: InferFn,
    cfg: BatchConfig,
}

impl ServeBuilder {
    pub fn new(infer: InferFn) -> Self {
        Self {
            infer,
            cfg: BatchConfig::default(),
        }
    }

    pub fn max_batch(mut self, n: usize) -> Self {
        self.cfg.max_batch = n.max(1);
        self
    }

    pub fn max_wait_ms(mut self, ms: u64) -> Self {
        self.cfg.max_wait = std::time::Duration::from_millis(ms);
        self
    }

    pub fn queue_depth(mut self, n: usize) -> Self {
        self.cfg.queue_depth = n.max(1);
        self
    }

    /// Build the axum router (without binding) — useful for tests
    /// that want to bind their own listener.
    pub fn into_router(self) -> Router {
        let batcher = Batcher::spawn(self.infer, self.cfg);
        let state = AppState {
            batcher,
            metrics: Arc::new(Metrics::default()),
        };
        Router::new()
            .route("/predict", post(predict))
            .route("/health", get(health))
            .route("/metrics", get(metrics))
            .with_state(state)
    }

    /// Bind + serve on `addr`. Returns when the server shuts down.
    pub async fn listen_http(self, addr: SocketAddr) -> Result<(), ServeError> {
        let app = self.into_router();
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app)
            .await
            .map_err(|e| ServeError::Serve(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn predict_endpoint_round_trips() {
        let infer: InferFn = Arc::new(|batch: Vec<serde_json::Value>| {
            batch.into_iter().map(|v| json!({"echo": v})).collect()
        });
        let app = ServeBuilder::new(infer).into_router();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let v: serde_json::Value = reqwest::Client::new()
            .post(format!("http://{addr}/predict"))
            .json(&json!({"inputs": {"x": 7}}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(v["outputs"]["echo"]["x"], 7);
    }

    #[tokio::test]
    async fn health_returns_ok_text() {
        let infer: InferFn = Arc::new(|batch| batch);
        let app = ServeBuilder::new(infer).into_router();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let body = reqwest::get(format!("http://{addr}/health"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert_eq!(body, "ok");
    }

    #[tokio::test]
    async fn metrics_endpoint_exposes_counters() {
        let infer: InferFn = Arc::new(|batch| batch);
        let app = ServeBuilder::new(infer).into_router();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        // One predict so the counter ticks.
        let _ = reqwest::Client::new()
            .post(format!("http://{addr}/predict"))
            .json(&json!({"inputs": null}))
            .send()
            .await
            .unwrap();
        let body = reqwest::get(format!("http://{addr}/metrics"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(body.contains("rustorch_serve_requests_total"));
    }
}
