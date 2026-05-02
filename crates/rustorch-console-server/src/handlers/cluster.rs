//! Cluster topology + per-GPU samples.
//!
//! v1 reads from a shared `ClusterProvider` (see `crate::cluster`)
//! so HTTP and SSE always see the same numbers. The provider picks
//! NVML when available (feature `nvml`), falls back to the
//! deterministic mock.

use crate::cluster::{self, ClusterProvider, GpuSample};
use crate::error::ApiResult;
use crate::state::AppState;
use axum::{extract::State, routing::get, Json, Router};
use serde::Serialize;

#[derive(Serialize, utoipa::ToSchema)]
pub struct ClusterHealth {
    pub healthy: bool,
    pub gpu_count: u32,
    pub busy_gpus: u32,
    pub message: &'static str,
}

#[utoipa::path(get, path = "/cluster/gpus", responses((status = 200, body = [GpuSample])))]
pub async fn gpus(State(s): State<AppState>) -> ApiResult<Json<Vec<GpuSample>>> {
    Ok(Json(s.cluster.samples()))
}

#[utoipa::path(get, path = "/cluster/health", responses((status = 200, body = ClusterHealth)))]
pub async fn health(State(s): State<AppState>) -> ApiResult<Json<ClusterHealth>> {
    let samples = s.cluster.samples();
    let busy = samples.iter().filter(|g| g.util > 5).count() as u32;
    Ok(Json(ClusterHealth {
        healthy: true,
        gpu_count: samples.len() as u32,
        busy_gpus: busy,
        message: "live samples — switch to NVML provider via --features nvml",
    }))
}

/// Convenience for tests — exposes the deterministic mock samples
/// without going through the trait object.
pub fn mock_samples() -> Vec<GpuSample> {
    cluster::MockClusterProvider::new().samples()
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/cluster/gpus", get(gpus))
        .route("/cluster/health", get(health))
}
