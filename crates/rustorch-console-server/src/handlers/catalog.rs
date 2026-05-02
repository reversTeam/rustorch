//! `/models`, `/datasets`, `/datasets/:name` — the read-only catalog
//! surface. v1 returns a hard-coded list aligned with the doc v0.7.2
//! examples (ResNet50, GPT-2, …). The real implementation will pull
//! from a registry (HuggingFace mirror or local YAML manifest) via
//! a `CatalogProvider` trait — kept private to this module so the
//! handler signatures stay stable when we swap the source.

use crate::error::{ApiError, ApiResult};
use axum::{extract::Path, routing::get, Json, Router};
use serde::Serialize;

#[derive(Serialize, utoipa::ToSchema, Clone)]
pub struct ModelEntry {
    pub name: String,
    pub family: &'static str,
    pub params_m: u32,
    pub default_dtype: &'static str,
    pub task: &'static str,
}

#[derive(Serialize, utoipa::ToSchema, Clone)]
pub struct DatasetSummary {
    pub name: String,
    pub task: &'static str,
    pub samples: u64,
    pub size_mb: u32,
    pub cached: bool,
}

#[derive(Serialize, utoipa::ToSchema, Clone)]
pub struct DatasetDetail {
    #[serde(flatten)]
    pub summary: DatasetSummary,
    pub splits: Vec<&'static str>,
    pub url: &'static str,
    pub sha256: &'static str,
}

#[utoipa::path(get, path = "/models", responses((status = 200, body = [ModelEntry])))]
pub async fn list_models() -> ApiResult<Json<Vec<ModelEntry>>> {
    Ok(Json(mock_models()))
}

#[utoipa::path(get, path = "/datasets", responses((status = 200, body = [DatasetSummary])))]
pub async fn list_datasets() -> ApiResult<Json<Vec<DatasetSummary>>> {
    Ok(Json(mock_datasets()))
}

#[utoipa::path(
    get, path = "/datasets/{name}",
    params(("name" = String, Path, description = "Dataset name (e.g. mnist, cifar10)")),
    responses((status = 200, body = DatasetDetail), (status = 404))
)]
pub async fn get_dataset(Path(name): Path<String>) -> ApiResult<Json<DatasetDetail>> {
    mock_dataset_detail(&name)
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("dataset {name}")))
}

fn mock_models() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            name: "resnet50".into(),
            family: "resnet",
            params_m: 25,
            default_dtype: "fp32",
            task: "image-classification",
        },
        ModelEntry {
            name: "resnet18".into(),
            family: "resnet",
            params_m: 11,
            default_dtype: "fp32",
            task: "image-classification",
        },
        ModelEntry {
            name: "gpt2-small".into(),
            family: "gpt",
            params_m: 124,
            default_dtype: "bf16",
            task: "text-generation",
        },
        ModelEntry {
            name: "vit-b-16".into(),
            family: "vit",
            params_m: 86,
            default_dtype: "fp32",
            task: "image-classification",
        },
    ]
}

fn mock_datasets() -> Vec<DatasetSummary> {
    vec![
        DatasetSummary {
            name: "mnist".into(),
            task: "image-classification",
            samples: 70_000,
            size_mb: 11,
            cached: true,
        },
        DatasetSummary {
            name: "cifar10".into(),
            task: "image-classification",
            samples: 60_000,
            size_mb: 163,
            cached: true,
        },
        DatasetSummary {
            name: "imagenet".into(),
            task: "image-classification",
            samples: 1_281_167,
            size_mb: 145_000,
            cached: false,
        },
    ]
}

fn mock_dataset_detail(name: &str) -> Option<DatasetDetail> {
    let summary = mock_datasets().into_iter().find(|d| d.name == name)?;
    Some(match name {
        "mnist" => DatasetDetail {
            summary,
            splits: vec!["train", "test"],
            url: "https://yann.lecun.com/exdb/mnist/",
            sha256: "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec6f0a3e1f0d29a",
        },
        "cifar10" => DatasetDetail {
            summary,
            splits: vec!["train", "test"],
            url: "https://www.cs.toronto.edu/~kriz/cifar.html",
            sha256: "6d958be5cb2f3c84e6f12e2e1c5b4a4f17d2f3d2c2c5e8f4f8f2f2f2f2f2f2f2",
        },
        "imagenet" => DatasetDetail {
            summary,
            splits: vec!["train", "val"],
            url: "https://image-net.org/",
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        },
        _ => return None,
    })
}

pub fn routes<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/models", get(list_models))
        .route("/datasets", get(list_datasets))
        .route("/datasets/:name", get(get_dataset))
}
