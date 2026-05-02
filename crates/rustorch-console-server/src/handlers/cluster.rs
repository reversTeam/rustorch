//! Cluster topology + per-GPU samples. v1 ships a deterministic mock
//! so the dashboard can be developed and snapshot-tested without a
//! GPU box. The NVML integration will plug in via a feature flag in
//! a follow-up commit (the response shape is already final).

use crate::error::ApiResult;
use axum::{routing::get, Json, Router};
use serde::Serialize;

#[derive(Serialize, utoipa::ToSchema, Clone)]
pub struct GpuSample {
    pub id: u32,
    /// Compute utilization 0-100.
    pub util: u32,
    /// VRAM used in MiB.
    pub vram_used_mb: u32,
    /// VRAM total in MiB.
    pub vram_total_mb: u32,
    /// Edge temperature in °C.
    pub temp_c: u32,
    /// Power draw in watts.
    pub power_w: u32,
    /// Run IDs currently bound to this GPU.
    pub runs: Vec<String>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct ClusterHealth {
    pub healthy: bool,
    pub gpu_count: u32,
    pub busy_gpus: u32,
    pub message: &'static str,
}

#[utoipa::path(get, path = "/cluster/gpus", responses((status = 200, body = [GpuSample])))]
pub async fn gpus() -> ApiResult<Json<Vec<GpuSample>>> {
    Ok(Json(mock_samples()))
}

#[utoipa::path(get, path = "/cluster/health", responses((status = 200, body = ClusterHealth)))]
pub async fn health() -> ApiResult<Json<ClusterHealth>> {
    let samples = mock_samples();
    let busy = samples.iter().filter(|g| g.util > 5).count() as u32;
    Ok(Json(ClusterHealth {
        healthy: true,
        gpu_count: samples.len() as u32,
        busy_gpus: busy,
        message: "mock provider — wire NVML in P2.4",
    }))
}

/// Deterministic samples so tests can assert on numbers. Real impl
/// will go through a `ClusterProvider` trait so we can swap NVML in
/// without changing handlers.
pub fn mock_samples() -> Vec<GpuSample> {
    vec![
        GpuSample {
            id: 0,
            util: 0,
            vram_used_mb: 256,
            vram_total_mb: 24_564,
            temp_c: 38,
            power_w: 60,
            runs: vec![],
        },
        GpuSample {
            id: 1,
            util: 0,
            vram_used_mb: 256,
            vram_total_mb: 24_564,
            temp_c: 39,
            power_w: 62,
            runs: vec![],
        },
    ]
}

pub fn routes<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/cluster/gpus", get(gpus))
        .route("/cluster/health", get(health))
}
