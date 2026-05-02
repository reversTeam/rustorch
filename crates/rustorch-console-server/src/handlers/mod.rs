//! HTTP handlers grouped by doc v0.7.2 §"Reference/Console API"
//! sections. Each module exposes a small `routes()` builder so the
//! top-level `router::build()` only does composition.

pub mod activity;
pub mod builder;
pub mod catalog;
pub mod cluster;
pub mod deploy;
pub mod fs;
pub mod me;
pub mod runs;
pub mod sse_routes;
pub mod sweeps;
