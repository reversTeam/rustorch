//! Backend HTTP/JSON-RPC + SSE pour la Console rustorch.
//!
//! The crate exposes a single `axum::Router` builder so it can be:
//!   * spawned by `main.rs` as `rustorch-console serve`,
//!   * embedded into integration tests via `axum::serve`,
//!   * eventually mounted under another service for a unified port.
//!
//! Everything not directly tied to HTTP — the SSE hub, the DB pool,
//! the error contract — is exposed as `pub` so plans P2.3 (sweeps),
//! P2.5 (logging sinks) and P2.8 (gRPC) can reuse the primitives
//! without re-implementing them.

pub mod auth;
pub mod db;
pub mod error;
pub mod handlers;
pub mod openapi;
pub mod router;
pub mod sse;
pub mod state;
