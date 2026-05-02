//! Cloneable application state — what every handler receives via
//! `axum::extract::State`. The DB pool and the SSE hub are both
//! clone-able themselves (Arc internally), so cloning `AppState` is
//! cheap.

use crate::sse::Hub;
use sqlx::SqlitePool;

#[derive(Clone)]
pub struct AppState {
    pub db: SqlitePool,
    pub hub: Hub,
}

impl AppState {
    pub fn new(db: SqlitePool, hub: Hub) -> Self {
        Self { db, hub }
    }
}
