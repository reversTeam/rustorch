//! End-to-end tests against a real `axum::serve` instance bound to
//! `127.0.0.1:0`. Each test starts its own server with an in-memory
//! SQLite DB so they're fully isolated and parallel-safe.

use rustorch_console_server::{
    auth::AuthConfig,
    cluster::MockClusterProvider,
    db::{self, NewCheckpoint},
    router,
    sse::Hub,
    state::AppState,
};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Spawn a fresh server on an ephemeral port. Returns the bound
/// address — the caller talks to it via `reqwest`.
async fn spawn(auth: AuthConfig) -> (SocketAddr, AppState) {
    let pool = db::connect(":memory:").await.expect("db");
    let state = AppState::new(pool, Hub::new(), Arc::new(MockClusterProvider::new()));
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

// ---- catalog --------------------------------------------------------

#[tokio::test]
async fn catalog_lists_models_and_datasets() {
    let (addr, _) = spawn(AuthConfig::default()).await;

    let models: Vec<serde_json::Value> = reqwest::get(format!("http://{addr}/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let names: Vec<&str> = models.iter().map(|m| m["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"resnet50"));
    assert!(names.contains(&"gpt2-small"));

    let datasets: Vec<serde_json::Value> = reqwest::get(format!("http://{addr}/datasets"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let names: Vec<&str> = datasets
        .iter()
        .map(|d| d["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"mnist"));
    assert!(names.contains(&"cifar10"));
}

#[tokio::test]
async fn catalog_dataset_detail_or_404() {
    let (addr, _) = spawn(AuthConfig::default()).await;

    let v: serde_json::Value = reqwest::get(format!("http://{addr}/datasets/mnist"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["name"], "mnist");
    assert!(v["splits"].as_array().unwrap().len() >= 2);

    let resp = reqwest::get(format!("http://{addr}/datasets/missing-one"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["code"], "NOT_FOUND");
}

// ---- /runs/:id read-only views ------------------------------------

/// Create a run via the lib (faster than HTTP) so each test has a
/// known id to query.
async fn seed_run(state: &AppState) -> String {
    let r = db::insert_run(
        &state.db,
        db::NewRun {
            title: Some("seed".into()),
            cfg_json: serde_json::json!({"lr": 1e-3, "code": "fn main() {}", "commit_sha": "abc123"}),
            sweep_id: None,
        },
    )
    .await
    .unwrap();
    r.id
}

#[tokio::test]
async fn runs_curves_returns_metrics_grouped_by_name() {
    let (addr, state) = spawn(AuthConfig::default()).await;
    let run_id = seed_run(&state).await;

    // Insert a few sample points across two metric names.
    for step in 0..3 {
        db::insert_metric(&state.db, &run_id, step, "loss", 1.0 - 0.1 * step as f64)
            .await
            .unwrap();
        db::insert_metric(&state.db, &run_id, step, "lr", 1e-3)
            .await
            .unwrap();
    }

    let v: serde_json::Value = reqwest::get(format!("http://{addr}/runs/{run_id}/curves"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let names: Vec<&str> = v["names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["loss", "lr"]);
    assert_eq!(v["points"].as_array().unwrap().len(), 6);

    // Missing run → 404.
    let resp = reqwest::get(format!("http://{addr}/runs/missing/curves"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn runs_hparams_echoes_cfg() {
    let (addr, state) = spawn(AuthConfig::default()).await;
    let run_id = seed_run(&state).await;

    let v: serde_json::Value = reqwest::get(format!("http://{addr}/runs/{run_id}/hparams"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["run_id"], run_id);
    assert_eq!(v["cfg"]["lr"], 1e-3);
}

#[tokio::test]
async fn runs_code_returns_recorded_source() {
    let (addr, state) = spawn(AuthConfig::default()).await;
    let run_id = seed_run(&state).await;

    let v: serde_json::Value = reqwest::get(format!("http://{addr}/runs/{run_id}/code"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["source"], "fn main() {}");
    assert_eq!(v["commit_sha"], "abc123");
}

#[tokio::test]
async fn runs_log_returns_placeholder_lines() {
    let (addr, state) = spawn(AuthConfig::default()).await;
    let run_id = seed_run(&state).await;

    let v: serde_json::Value = reqwest::get(format!("http://{addr}/runs/{run_id}/log?tail=50"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["run_id"], run_id);
    let lines = v["lines"].as_array().unwrap();
    assert!(!lines.is_empty());
    assert_eq!(lines[0]["level"], "info");
}

#[tokio::test]
async fn runs_checkpoints_lists_in_step_desc() {
    let (addr, state) = spawn(AuthConfig::default()).await;
    let run_id = seed_run(&state).await;

    for step in [0, 5, 10] {
        db::insert_checkpoint(
            &state.db,
            NewCheckpoint {
                run_id: run_id.clone(),
                path: format!("ckpt-{step}.safetensors"),
                step,
                metrics: serde_json::json!({"val_acc": 0.5 + 0.05 * step as f64}),
            },
        )
        .await
        .unwrap();
    }

    let arr: Vec<serde_json::Value> =
        reqwest::get(format!("http://{addr}/runs/{run_id}/checkpoints"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
    assert_eq!(arr.len(), 3);
    // Sorted by step DESC — newest first.
    assert_eq!(arr[0]["step"], 10);
    assert_eq!(arr[2]["step"], 0);
    assert!(arr[0]["metrics"]["val_acc"].as_f64().unwrap() > 0.9);
}

#[tokio::test]
async fn runs_artifacts_mirror_checkpoints() {
    let (addr, state) = spawn(AuthConfig::default()).await;
    let run_id = seed_run(&state).await;

    db::insert_checkpoint(
        &state.db,
        NewCheckpoint {
            run_id: run_id.clone(),
            path: "best.safetensors".into(),
            step: 1,
            metrics: serde_json::json!({}),
        },
    )
    .await
    .unwrap();

    let arr: Vec<serde_json::Value> =
        reqwest::get(format!("http://{addr}/runs/{run_id}/artifacts"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["kind"], "checkpoint");
    assert_eq!(arr[0]["path"], "best.safetensors");
}

#[tokio::test]
async fn runs_system_returns_mock_samples() {
    let (addr, state) = spawn(AuthConfig::default()).await;
    let run_id = seed_run(&state).await;

    let v: serde_json::Value = reqwest::get(format!("http://{addr}/runs/{run_id}/system"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["run_id"], run_id);
    assert!(!v["samples"].as_array().unwrap().is_empty());
    assert_eq!(v["samples"][0]["gpu_id"], 0);
}

#[tokio::test]
async fn runs_fork_clones_cfg_and_tags_lineage() {
    let (addr, state) = spawn(AuthConfig::default()).await;
    let parent_id = seed_run(&state).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("http://{addr}/runs/{parent_id}/fork"))
        .json(&json!({
            "title": "experiment-A",
            "overrides": {"lr": 3e-4, "optim": "lion"},
            "resume_from": "ckpt-10.safetensors"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let child: serde_json::Value = resp.json().await.unwrap();

    // Child is queued, has its own ULID, and inherited the parent's cfg
    // with the overrides applied.
    assert_eq!(child["status"], "queued");
    assert_eq!(child["title"], "experiment-A");
    let cfg: serde_json::Value = serde_json::from_str(child["cfg_json"].as_str().unwrap()).unwrap();
    assert_eq!(cfg["lr"], 3e-4);
    assert_eq!(cfg["optim"], "lion");
    assert_eq!(cfg["parent_run_id"], parent_id);
    assert_eq!(cfg["resume_from"], "ckpt-10.safetensors");

    // Parent and child are different rows.
    assert_ne!(child["id"].as_str().unwrap(), parent_id);

    // Forking a missing run → 404.
    let resp = client
        .post(format!("http://{addr}/runs/no-such-run/fork"))
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

// ---- SSE replay + scale -----------------------------------------------

#[tokio::test]
async fn sse_replays_via_last_event_id_header() {
    let (addr, state) = spawn(AuthConfig::default()).await;

    // Burn three events into the topic before any subscriber exists.
    for v in 1..=3 {
        state
            .hub
            .publish("runs.changed", "runs.changed", &json!({"v": v}));
    }

    // Reconnect with `Last-Event-ID: 1` → server should replay id=2 and id=3
    // before going live, then we'll publish a 4th to confirm the live path.
    use eventsource_stream::Eventsource;
    use futures::StreamExt;
    use std::time::Duration;
    use tokio::time::timeout;

    let recv = tokio::spawn(async move {
        let resp = reqwest::Client::new()
            .get(format!("http://{addr}/sse/runs.changed"))
            .header("Last-Event-ID", "1")
            .send()
            .await
            .unwrap();
        let mut stream = resp.bytes_stream().eventsource();
        let mut got = Vec::new();
        for _ in 0..3 {
            let item = timeout(Duration::from_secs(2), stream.next())
                .await
                .ok()
                .flatten()
                .ok_or("no event")?
                .map_err(|e| format!("sse: {e}"))?;
            got.push((item.id.clone(), item.data.clone()));
        }
        Ok::<_, String>(got)
    });

    // Give the subscriber a moment to subscribe before firing the live event.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    state
        .hub
        .publish("runs.changed", "runs.changed", &json!({"v": 4}));

    let got = recv.await.unwrap().expect("sse delivery");
    let ids: Vec<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
    // Replay yields id=2,3 ; then id=4 from the live channel.
    assert_eq!(ids, vec!["2", "3", "4"]);
}

#[tokio::test]
async fn sse_handles_100_concurrent_subscribers() {
    use eventsource_stream::Eventsource;
    use futures::StreamExt;
    use std::time::Duration;
    use tokio::time::timeout;

    let (addr, state) = spawn(AuthConfig::default()).await;

    // Spawn 100 tasks, each opens an SSE stream and waits for the
    // first event. They all subscribe to the same topic before we
    // publish anything.
    let n = 100;
    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        handles.push(tokio::spawn(async move {
            let resp = reqwest::Client::new()
                .get(format!("http://{addr}/sse/runs.changed"))
                .send()
                .await
                .unwrap();
            let mut stream = resp.bytes_stream().eventsource();
            let item = timeout(Duration::from_secs(5), stream.next())
                .await
                .ok()
                .flatten()
                .ok_or("no event")?
                .map_err(|e| format!("sse: {e}"))?;
            Ok::<_, String>(item.data)
        }));
    }

    // Wait until the broadcast channel actually has 100 receivers
    // hooked in (subscribe happens lazily on the GET handler).
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while state.hub.subscriber_count("runs.changed") < n && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        state.hub.subscriber_count("runs.changed"),
        n,
        "all 100 subscribers should be hooked in before publish"
    );

    state
        .hub
        .publish("runs.changed", "runs.changed", &json!({"id": "fanout"}));

    // Every subscriber must see the event.
    for h in handles {
        let payload = h.await.unwrap().expect("sub got the event");
        let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(v["id"], "fanout");
    }
}

#[tokio::test]
async fn sse_run_checkpoint_topic_fires_on_save() {
    use eventsource_stream::Eventsource;
    use futures::StreamExt;
    use std::time::Duration;
    use tokio::time::timeout;

    let (addr, state) = spawn(AuthConfig::default()).await;
    let run_id = seed_run(&state).await;

    // Subscribe to the per-run checkpoint topic before we POST.
    let sse_url = format!("http://{addr}/sse/runs/{run_id}/checkpoint");
    let recv = tokio::spawn(async move {
        let resp = reqwest::Client::new().get(&sse_url).send().await.unwrap();
        let mut stream = resp.bytes_stream().eventsource();
        let item = timeout(Duration::from_secs(2), stream.next())
            .await
            .ok()
            .flatten()
            .ok_or("no event")?
            .map_err(|e| format!("sse: {e}"))?;
        Ok::<_, String>(item.data)
    });

    tokio::time::sleep(Duration::from_millis(150)).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/runs/{run_id}/checkpoints"))
        .json(&json!({
            "path": "epoch_5.safetensors",
            "step": 500,
            "metrics": {"val_acc": 0.91},
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let payload = recv.await.unwrap().expect("sse delivery");
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(v["run_id"], run_id);
    assert_eq!(v["path"], "epoch_5.safetensors");
    assert_eq!(v["step"], 500);
    assert_eq!(v["metrics"]["val_acc"], 0.91);
}

#[tokio::test]
async fn sse_event_id_field_is_set() {
    use eventsource_stream::Eventsource;
    use futures::StreamExt;
    use std::time::Duration;
    use tokio::time::timeout;

    let (addr, state) = spawn(AuthConfig::default()).await;

    let recv = tokio::spawn(async move {
        let resp = reqwest::Client::new()
            .get(format!("http://{addr}/sse/runs.changed"))
            .send()
            .await
            .unwrap();
        let mut stream = resp.bytes_stream().eventsource();
        let item = timeout(Duration::from_secs(2), stream.next())
            .await
            .ok()
            .flatten()
            .ok_or("no event")?
            .map_err(|e| format!("sse: {e}"))?;
        Ok::<_, String>(item.id)
    });

    tokio::time::sleep(Duration::from_millis(100)).await;
    state
        .hub
        .publish("runs.changed", "runs.changed", &json!({"v": 1}));

    let id = recv.await.unwrap().expect("sse delivery");
    // First event on a fresh topic always has id=1.
    assert_eq!(id, "1");
}

#[tokio::test]
async fn cluster_tick_task_streams_via_sse() {
    use eventsource_stream::Eventsource;
    use futures::StreamExt;
    use rustorch_console_server::cluster::{spawn_tick_task, TICK_TOPIC};
    use std::time::Duration;
    use tokio::time::timeout;

    let (addr, state) = spawn(AuthConfig::default()).await;

    // Spin up a fast 50ms tick so the test doesn't sit on its hands.
    let _handle = spawn_tick_task(
        state.hub.clone(),
        state.cluster.clone(),
        Duration::from_millis(50),
    );

    // Subscribe via the public SSE endpoint and grab the first
    // delivered event.
    let url = format!("http://{addr}/sse/cluster.tick");
    let recv = tokio::spawn(async move {
        let resp = reqwest::Client::new().get(&url).send().await.unwrap();
        let mut stream = resp.bytes_stream().eventsource();
        let item = timeout(Duration::from_secs(2), stream.next())
            .await
            .ok()
            .flatten()
            .ok_or("no event")?
            .map_err(|e| format!("sse: {e}"))?;
        Ok::<_, String>(item.data)
    });

    let payload = recv.await.unwrap().expect("cluster.tick delivery");
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    let gpus = v["gpus"].as_array().expect("gpus array");
    assert_eq!(gpus.len(), 2);
    assert!(gpus[0]["util"].as_u64().unwrap() <= 100);
    assert_eq!(gpus[0]["id"], 0);
    assert!(state.hub.ring_len(TICK_TOPIC) >= 1);
}

#[tokio::test]
async fn checkpoint_against_missing_run_is_404() {
    let (_addr, state) = spawn(AuthConfig::default()).await;
    let err = db::insert_checkpoint(
        &state.db,
        NewCheckpoint {
            run_id: "no-such-run".into(),
            path: "p".into(),
            step: 0,
            metrics: serde_json::json!({}),
        },
    )
    .await
    .unwrap_err();
    let s = format!("{err:?}");
    assert!(s.contains("NotFound"), "expected NotFound, got {s}");
}
