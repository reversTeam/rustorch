//! Cloneable application state — what every handler receives via
//! `axum::extract::State`. The DB pool, SSE hub and cluster
//! provider are all clone-able themselves (Arc internally), so
//! cloning `AppState` is cheap.

use crate::cluster::ClusterProvider;
use crate::sse::Hub;
use sqlx::SqlitePool;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub db: SqlitePool,
    pub hub: Hub,
    pub cluster: Arc<dyn ClusterProvider>,
}

impl AppState {
    pub fn new(db: SqlitePool, hub: Hub, cluster: Arc<dyn ClusterProvider>) -> Self {
        Self { db, hub, cluster }
    }
}
