//! `rustorch-console serve` — the binary that backs the frontend.
//! Reads CLI flags + env, opens the SQLite pool, mounts the axum
//! router and listens.

use clap::Parser;
use rustorch_console_server::{
    auth::AuthConfig,
    cluster::{self, default_provider, spawn_tick_task},
    db, router,
    sse::Hub,
    state::AppState,
};
use std::net::SocketAddr;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "rustorch-console", about = "rustorch console backend")]
struct Cli {
    /// TCP address to bind on.
    #[arg(long, default_value = "127.0.0.1:8080", env = "RUSTORCH_CONSOLE_ADDR")]
    addr: SocketAddr,

    /// SQLite path. Use `:memory:` to keep state ephemeral.
    #[arg(long, env = "RUSTORCH_CONSOLE_DB")]
    db: Option<String>,
}

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[tokio::main]
async fn main() -> Result<(), DynError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,sqlx::query=warn".into()),
        )
        .init();

    let cli = Cli::parse();
    let db_path = cli
        .db
        .clone()
        .unwrap_or_else(|| db::default_db_path().to_string_lossy().into_owned());

    let pool = db::connect(&db_path)
        .await
        .map_err(|e| format!("db connect: {e}"))?;
    let hub = Hub::new();
    let provider = default_provider();
    let state = AppState::new(pool, hub.clone(), provider.clone());
    let auth = AuthConfig::from_env();
    let app = router::build(state, auth);

    // Background producers: 1Hz NVML/mock cluster.tick.
    let _tick_handle = spawn_tick_task(hub.clone(), provider.clone(), Duration::from_secs(1));
    let _ = cluster::TICK_TOPIC;

    let listener = tokio::net::TcpListener::bind(cli.addr).await?;
    tracing::info!(addr = %cli.addr, db = %db_path, "rustorch-console listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("ctrl-c received, shutting down");
}
