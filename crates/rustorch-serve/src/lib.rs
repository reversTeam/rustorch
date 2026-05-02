//! HTTP serving runtime with dynamic batching. Aligned with doc
//! v0.7.2 §`/Recipes/Inference & serving`.
//!
//! ### Surface
//!
//! ```rust,ignore
//! use rustorch_serve::ServeBuilder;
//!
//! let infer = std::sync::Arc::new(|batch: Vec<serde_json::Value>| {
//!     // Real impl runs the model on `batch`; here we echo.
//!     batch
//! });
//! ServeBuilder::new(infer)
//!     .max_batch(32)
//!     .max_wait_ms(20)
//!     .listen_http("0.0.0.0:8000")
//!     .await?;
//! ```
//!
//! ### Why HTTP first
//!
//! HTTP `/predict` is the universal interface — debuggable with curl,
//! works through any proxy. gRPC `Predict` / `PredictStream` /
//! `PredictBatch` arrive in a follow-up commit (needs a tonic build
//! step + protoc on the toolchain).
//!
//! ### Dynamic batching
//!
//! Each incoming request is enqueued via a tokio mpsc channel.
//! A single batcher task drains up to `max_batch` requests, waiting
//! at most `max_wait_ms` for the queue to fill, then runs them
//! through the inference closure in one shot. Per-request `oneshot`
//! channels carry the response back. The two knobs translate
//! directly to the latency / throughput trade-off documented in the
//! plan.

mod autoscale;
mod batcher;
mod cuda_graphs;
mod server;

pub use autoscale::{
    render_docker_compose, render_k8s_hpa_manifest, AutoscaleController, AutoscaleTarget,
    LoadSnapshot, ScaleDecision,
};
pub use batcher::{BatchConfig, Batcher, InferFn};
pub use cuda_graphs::{graph_aware_infer, GraphKey, GraphRunner, MockGraphRunner};
pub use server::{ServeBuilder, ServeError};
