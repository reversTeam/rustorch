//! Cloneable application state — what every handler receives via
//! `axum::extract::State`. The DB pool, SSE hub, cluster provider
//! and workspace path are all clone-able themselves (Arc internally
//! or trivially `Clone`), so cloning `AppState` is cheap.

use crate::cluster::ClusterProvider;
use crate::sse::Hub;
use sqlx::SqlitePool;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub db: SqlitePool,
    pub hub: Hub,
    pub cluster: Arc<dyn ClusterProvider>,
    /// Sandbox root for the FS endpoints + cargo runner. Defaults
    /// to the current working directory at startup.
    pub workspace_dir: PathBuf,
}

impl AppState {
    pub fn new(db: SqlitePool, hub: Hub, cluster: Arc<dyn ClusterProvider>) -> Self {
        Self {
            db,
            hub,
            cluster,
            workspace_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    pub fn with_workspace(mut self, dir: PathBuf) -> Self {
        self.workspace_dir = dir;
        self
    }
}
