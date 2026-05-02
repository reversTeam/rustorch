//! `/graph/*` — Model Builder backend. Stores the working DAG as a
//! single JSON blob in SQLite (one row, key = `current`) so the
//! frontend can save / load between sessions. Presets are
//! hard-coded for v1 — they'll move into a YAML registry once we
//! settle on the right schema.

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use axum::{extract::State, routing::get, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, utoipa::ToSchema)]
pub struct GraphDoc {
    /// Free-form node array — the editor owns the schema, the
    /// server just persists.
    pub nodes: serde_json::Value,
    /// Edges (wires) between nodes.
    pub edges: serde_json::Value,
    /// Optional metadata bag (title, description, …).
    #[serde(default)]
    pub meta: serde_json::Value,
}

const GRAPH_KEY: &str = "current";

#[utoipa::path(get, path = "/graph/current", responses((status = 200, body = GraphDoc)))]
pub async fn get_current(State(s): State<AppState>) -> ApiResult<Json<GraphDoc>> {
    let row: Option<(String,)> = sqlx::query_as("SELECT json FROM kv WHERE key = ?")
        .bind(GRAPH_KEY)
        .fetch_optional(&s.db)
        .await?;
    let doc = match row {
        Some((j,)) => serde_json::from_str(&j)?,
        None => GraphDoc {
            nodes: json!([]),
            edges: json!([]),
            meta: json!({}),
        },
    };
    Ok(Json(doc))
}

#[utoipa::path(
    put, path = "/graph/current",
    request_body = GraphDoc,
    responses((status = 200, body = GraphDoc))
)]
pub async fn put_current(
    State(s): State<AppState>,
    Json(doc): Json<GraphDoc>,
) -> ApiResult<Json<GraphDoc>> {
    let j = serde_json::to_string(&doc)?;
    sqlx::query("INSERT OR REPLACE INTO kv (key, json) VALUES (?, ?)")
        .bind(GRAPH_KEY)
        .bind(&j)
        .execute(&s.db)
        .await?;
    Ok(Json(doc))
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct PresetEntry {
    pub name: &'static str,
    pub description: &'static str,
}

#[utoipa::path(get, path = "/graph/presets", responses((status = 200, body = [PresetEntry])))]
pub async fn presets() -> ApiResult<Json<Vec<PresetEntry>>> {
    Ok(Json(vec![
        PresetEntry {
            name: "resnet50",
            description: "ResNet-50 image classification (ImageNet 1000-class).",
        },
        PresetEntry {
            name: "resnet18",
            description: "ResNet-18 — smaller, fits on consumer GPUs.",
        },
        PresetEntry {
            name: "vit-b-16",
            description: "Vision Transformer base, patch 16.",
        },
        PresetEntry {
            name: "gpt2-small",
            description: "GPT-2 small — 124M params, text generation.",
        },
        PresetEntry {
            name: "unet",
            description: "U-Net for semantic segmentation.",
        },
        PresetEntry {
            name: "dcgan",
            description: "Deep Convolutional GAN — image generation.",
        },
    ]))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/graph/current", get(get_current).put(put_current))
        .route("/graph/presets", get(presets))
}

/// Used by the trait-bound check.
#[allow(dead_code)]
fn _err_check() -> ApiError {
    ApiError::Internal("unreachable".into())
}
