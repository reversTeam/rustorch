//! Shared test infrastructure for rustorch-cpu.
//!
//! Currently exposes [`broadcast`] — a test grid covering every relevant
//! shape pair pattern for binary elementwise ops. Used by every binary
//! op integration test to keep the parity surface honest.

#![allow(dead_code)] // accessed by integration tests via `mod common;`

pub mod broadcast;
