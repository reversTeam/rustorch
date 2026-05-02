//! HTTP handlers grouped by doc v0.7.2 §"Reference/Console API"
//! sections. Each module exposes a small `routes()` builder so the
//! top-level `router::build()` only does composition.

pub mod cluster;
pub mod me;
pub mod runs;
pub mod sse_routes;
