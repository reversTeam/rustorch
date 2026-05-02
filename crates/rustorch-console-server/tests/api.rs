//! End-to-end tests against a real `axum::serve` instance bound to
//! `127.0.0.1:0`. Each test starts its own server with an in-memory
//! SQLite DB so they're fully isolated and parallel-safe.

use rustorch_console_server::{auth::AuthConfig, db, router, sse::Hub, state::AppState};
use serde_json::json;
use std::net::SocketAddr;
use tokio::net::TcpListener;

/// Spawn a fresh server on an ephemeral port. Returns the bound
/// address — the caller talks to it via `reqwest`.
async fn spawn(auth: AuthConfig) -> (SocketAddr, AppState) {
    let pool = db::connect(":memory:").await.expect("db");
    let state = AppState::new(pool, Hub::new());
    let app = router::build(state.clone(), auth);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, state)
}

#[tokio::test]
async fn health_returns_ok() {
    let (addr, _) = spawn(AuthConfig::default()).await;
    let v: serde_json::Value = reqwest::get(format!("http://{addr}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["status"], "ok");
}

#[tokio::test]
async fn me_and_workspace_match_contract() {
    let (addr, _) = spawn(AuthConfig::default()).await;
    let me: serde_json::Value = reqwest::get(format!("http://{addr}/me"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(me["workspace"], "default");
    let ws: serde_json::Value = reqwest::get(format!("http://{addr}/workspace"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(ws["server_version"].is_string());
}

#[tokio::test]
async fn cluster_endpoints_return_mock_samples() {
    let (addr, _) = spawn(AuthConfig::default()).await;
    let gpus: Vec<serde_json::Value> = reqwest::get(format!("http://{addr}/cluster/gpus"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(gpus.len(), 2);
    assert!(gpus[0]["vram_total_mb"].as_u64().unwrap() > 0);

    let h: serde_json::Value = reqwest::get(format!("http://{addr}/cluster/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(h["healthy"], true);
    assert_eq!(h["gpu_count"], 2);
}

#[tokio::test]
async fn run_lifecycle_full_path() {
    let (addr, _) = spawn(AuthConfig::default()).await;
    let client = reqwest::Client::new();

    // 1) Create
    let created: serde_json::Value = client
        .post(format!("http://{addr}/runs"))
        .json(&json!({"title": "first", "cfg": {"lr": 1e-3}}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = created["id"].as_str().unwrap().to_string();
    assert_eq!(created["status"], "queued");

    // 2) List with status filter
    let listed: serde_json::Value = client
        .get(format!("http://{addr}/runs?status=queued"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listed["total"], 1);
    assert_eq!(listed["items"][0]["id"], id);

    // 3) Start (queued → running)
    let started: serde_json::Value = client
        .post(format!("http://{addr}/runs/{id}/start"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(started["status"], "running");

    // 4) Pause (running → paused)
    let paused: serde_json::Value = client
        .post(format!("http://{addr}/runs/{id}/pause"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(paused["status"], "paused");

    // 5) Resume (paused → running)
    let resumed: serde_json::Value = client
        .post(format!("http://{addr}/runs/{id}/resume"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resumed["status"], "running");

    // 6) Stop (running → cancelled)
    let stopped: serde_json::Value = client
        .post(format!("http://{addr}/runs/{id}/stop"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(stopped["status"], "cancelled");
}

#[tokio::test]
async fn invalid_transition_returns_409() {
    let (addr, _) = spawn(AuthConfig::default()).await;
    let client = reqwest::Client::new();

    let created: serde_json::Value = client
        .post(format!("http://{addr}/runs"))
        .json(&json!({"cfg": {}}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = created["id"].as_str().unwrap();

    // queued cannot directly pause — must go through start first.
    let resp = client
        .post(format!("http://{addr}/runs/{id}/pause"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 409);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["code"], "CONFLICT");
}

#[tokio::test]
async fn invalid_cfg_returns_400() {
    let (addr, _) = spawn(AuthConfig::default()).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/runs"))
        .json(&json!({"cfg": "not-an-object"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["code"], "INVALID_ARG");
}

#[tokio::test]
async fn missing_run_returns_404() {
    let (addr, _) = spawn(AuthConfig::default()).await;
    let resp = reqwest::get(format!("http://{addr}/runs/missing-id"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["code"], "NOT_FOUND");
}

#[tokio::test]
async fn bearer_auth_blocks_unauthenticated() {
    let auth = AuthConfig {
        token: Some("secret".into()),
    };
    let (addr, _) = spawn(auth).await;

    // No header → 401.
    let resp = reqwest::get(format!("http://{addr}/me")).await.unwrap();
    assert_eq!(resp.status(), 401);

    // Wrong token → 401.
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/me"))
        .header("Authorization", "Bearer wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    // Right token → 200.
    let resp = client
        .get(format!("http://{addr}/me"))
        .header("Authorization", "Bearer secret")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Health is always reachable.
    let resp = reqwest::get(format!("http://{addr}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn openapi_spec_is_served() {
    let (addr, _) = spawn(AuthConfig::default()).await;
    let v: serde_json::Value = reqwest::get(format!("http://{addr}/openapi.json"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["openapi"].as_str().unwrap_or(""), "3.1.0");
    assert_eq!(v["info"]["title"], "rustorch console API");
    // At least one of our paths must be there.
    assert!(v["paths"]["/runs"].is_object());
    assert!(v["paths"]["/cluster/gpus"].is_object());
}

#[tokio::test]
async fn sse_runs_changed_emits_lifecycle_event() {
    use eventsource_stream::Eventsource;
    use futures::StreamExt;
    use std::time::Duration;
    use tokio::time::timeout;

    let (addr, _) = spawn(AuthConfig::default()).await;

    // Start an SSE subscription on a background task.
    let sse_addr = addr;
    let recv_handle = tokio::spawn(async move {
        let resp = reqwest::Client::new()
            .get(format!("http://{sse_addr}/sse/runs.changed"))
            .send()
            .await
            .unwrap();
        let mut stream = resp.bytes_stream().eventsource();
        // Read the first non-keep-alive event; bail after 2s.
        let item = timeout(Duration::from_secs(2), stream.next())
            .await
            .ok()
            .flatten()
            .ok_or("no event")?
            .map_err(|e| format!("sse: {e}"))?;
        Ok::<_, String>(item.data)
    });

    // Give the subscriber a beat to register before firing the event.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let created: serde_json::Value = client
        .post(format!("http://{addr}/runs"))
        .json(&json!({"cfg": {}}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = created["id"].as_str().unwrap().to_string();

    let payload = recv_handle.await.unwrap().expect("sse delivery");
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(v["id"], id);
    assert_eq!(v["status"], "queued");
}
