//! OpenAPI 3.1 schema, served at `/openapi.json`. We deliberately
//! list paths and components by hand here rather than via the
//! `#[derive(OpenApi)]` `paths(...)` attribute so adding a new
//! handler is a one-liner change.

use crate::db;
use crate::handlers::{catalog, cluster, me, runs};
use utoipa::OpenApi;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "rustorch console API",
        description = "HTTP/JSON-RPC backend for the rustorch console UI (doc v0.7.2 §Reference/Console API).",
        version = "0.0.1",
    ),
    paths(
        me::me,
        me::workspace,
        cluster::gpus,
        cluster::health,
        catalog::list_models,
        catalog::list_datasets,
        catalog::get_dataset,
        runs::list,
        runs::get_one,
        runs::create,
        runs::start,
        runs::pause,
        runs::resume,
        runs::stop,
        runs::fork,
        runs::curves,
        runs::hparams,
        runs::code,
        runs::log,
        runs::checkpoints,
        runs::artifacts,
        runs::system,
    ),
    components(schemas(
        me::MeResponse,
        me::WorkspaceResponse,
        cluster::GpuSample,
        cluster::ClusterHealth,
        catalog::ModelEntry,
        catalog::DatasetSummary,
        catalog::DatasetDetail,
        runs::ListResponse,
        runs::CreateBody,
        runs::ForkBody,
        runs::CurvesResponse,
        runs::HparamsResponse,
        runs::CodeResponse,
        runs::LogLine,
        runs::LogResponse,
        runs::ArtifactEntry,
        runs::SystemSample,
        runs::SystemResponse,
        db::Run,
        db::RunSummary,
        db::RunStatus,
        db::MetricPoint,
        db::Checkpoint,
        db::ActivityEvent,
    )),
    tags(
        (name = "me",       description = "Identity & workspace metadata"),
        (name = "cluster",  description = "Cluster topology + GPU samples"),
        (name = "catalog",  description = "Model & dataset registry"),
        (name = "runs",     description = "Run lifecycle + read-only views"),
    )
)]
pub struct ApiDoc;
