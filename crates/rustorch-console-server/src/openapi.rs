//! OpenAPI 3.1 schema, served at `/openapi.json`. We deliberately
//! list paths and components by hand here rather than via the
//! `#[derive(OpenApi)]` `paths(...)` attribute so adding a new
//! handler is a one-liner change.

use crate::db;
use crate::handlers::{cluster, me, runs};
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
        runs::list,
        runs::get_one,
        runs::create,
        runs::start,
        runs::pause,
        runs::resume,
        runs::stop,
    ),
    components(schemas(
        me::MeResponse,
        me::WorkspaceResponse,
        cluster::GpuSample,
        cluster::ClusterHealth,
        runs::ListResponse,
        runs::CreateBody,
        db::Run,
        db::RunSummary,
        db::RunStatus,
    )),
    tags(
        (name = "me",      description = "Identity & workspace metadata"),
        (name = "cluster", description = "Cluster topology + GPU samples"),
        (name = "runs",    description = "Run lifecycle + listing"),
    )
)]
pub struct ApiDoc;
