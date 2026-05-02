//! In-process runner registry. Same conceptual shape as the
//! eventual gRPC server — `Registry::register` returns a
//! `RunnerSession` whose `send_event` / `recv_command` match the
//! bidirectional gRPC stream behaviour.
//!
//! Used by the Console backend's tests + by single-process
//! deployments where the Console and Runner share an address space.

use crate::types::*;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("runner {0} not found")]
    NotFound(String),
    #[error("runner {0} already disconnected")]
    Disconnected(String),
}

/// One live runner — what the runner-side code drives.
///
/// * `events_tx`  — the runner pushes status / metrics / etc.
///   The registry's forwarder picks them up via the matched
///   receiver stored as `events_rx_holder`.
/// * `commands_rx` — the runner awaits commands here. Senders live
///   in the registry's per-runner entry.
pub struct RunnerSession {
    pub runner_id: String,
    pub run_id: String,
    pub events_tx: mpsc::Sender<RunnerEvent>,
    pub commands_rx: mpsc::Receiver<ControlCommand>,
}

impl RunnerSession {
    /// Push an event upstream.
    pub async fn send_event(&self, ev: RunnerEvent) -> Result<(), RegistryError> {
        self.events_tx
            .send(ev)
            .await
            .map_err(|_| RegistryError::Disconnected(self.runner_id.clone()))
    }

    /// Receive the next command from the console.
    pub async fn next_command(&mut self) -> Option<ControlCommand> {
        self.commands_rx.recv().await
    }
}

/// Bookkeeping from the registry's POV — the matching tx for
/// commands + rx for events.
struct RunnerEntry {
    run_id: String,
    commands_tx: mpsc::Sender<ControlCommand>,
    /// Holding the events_rx here keeps the channel open until the
    /// console explicitly forgets the runner. A real impl would
    /// drain it into the SSE hub via a background task; this lib
    /// exposes `take_events_rx` for the caller to wire.
    events_rx_holder: parking_lot::Mutex<Option<mpsc::Receiver<RunnerEvent>>>,
}

/// Cloneable handle to the registry. Internally `Arc<RwLock<HashMap>>`.
#[derive(Clone, Default)]
pub struct Registry {
    inner: Arc<RwLock<HashMap<String, RunnerEntry>>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a runner. Returns the live session the caller drives
    /// — typically inside a background task that loops on
    /// `next_event` and forwards into the Console DB / SSE hub.
    pub fn register(&self, req: RegisterRequest) -> (RegisterResponse, RunnerSession) {
        let runner_id = ulid::Ulid::new().to_string();
        let session_token = ulid::Ulid::new().to_string();
        let (events_tx, events_rx) = mpsc::channel(256);
        let (commands_tx, commands_rx) = mpsc::channel(64);

        let resp = RegisterResponse {
            runner_id: runner_id.clone(),
            session_token,
        };
        let session = RunnerSession {
            runner_id: runner_id.clone(),
            run_id: req.run_id.clone(),
            events_tx,
            commands_rx,
        };
        self.inner.write().insert(
            runner_id,
            RunnerEntry {
                run_id: req.run_id,
                commands_tx,
                events_rx_holder: parking_lot::Mutex::new(Some(events_rx)),
            },
        );
        (resp, session)
    }

    /// Take the inbound events receiver for `runner_id`. Returns
    /// `None` if the runner doesn't exist or the receiver was
    /// already taken (a forwarder normally takes it once).
    pub fn take_events_rx(&self, runner_id: &str) -> Option<mpsc::Receiver<RunnerEvent>> {
        self.inner
            .read()
            .get(runner_id)
            .and_then(|e| e.events_rx_holder.lock().take())
    }

    /// Send a control command to a specific runner.
    pub async fn send_command(
        &self,
        runner_id: &str,
        cmd: ControlCommand,
    ) -> Result<(), RegistryError> {
        let tx = self
            .inner
            .read()
            .get(runner_id)
            .map(|e| e.commands_tx.clone())
            .ok_or_else(|| RegistryError::NotFound(runner_id.to_string()))?;
        tx.send(cmd)
            .await
            .map_err(|_| RegistryError::Disconnected(runner_id.to_string()))
    }

    /// Currently-known runner ids.
    pub fn runner_ids(&self) -> Vec<String> {
        self.inner.read().keys().cloned().collect()
    }

    /// Find the runner currently bound to a run.
    pub fn runner_for_run(&self, run_id: &str) -> Option<String> {
        self.inner
            .read()
            .iter()
            .find(|(_, e)| e.run_id == run_id)
            .map(|(id, _)| id.clone())
    }

    /// Forget a runner (called when the gRPC stream closes).
    pub fn forget(&self, runner_id: &str) -> bool {
        self.inner.write().remove(runner_id).is_some()
    }

    /// Number of currently-registered runners.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(run_id: &str) -> RegisterRequest {
        RegisterRequest {
            run_id: run_id.into(),
            gpu_count: 1,
            runner_version: "0.0.1".into(),
        }
    }

    #[tokio::test]
    async fn register_returns_unique_runner_id() {
        let r = Registry::new();
        let (a, _) = r.register(req("run-a"));
        let (b, _) = r.register(req("run-b"));
        assert_ne!(a.runner_id, b.runner_id);
        assert_eq!(r.len(), 2);
    }

    #[tokio::test]
    async fn commands_flow_to_session() {
        let r = Registry::new();
        let (resp, mut session) = r.register(req("run-1"));
        r.send_command(&resp.runner_id, ControlCommand::Pause)
            .await
            .unwrap();
        let cmd = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            session.next_command(),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(cmd, ControlCommand::Pause);
    }

    #[tokio::test]
    async fn events_flow_back_via_take_events_rx() {
        let r = Registry::new();
        let (resp, session) = r.register(req("run-2"));
        let mut events_rx = r.take_events_rx(&resp.runner_id).expect("rx");
        session
            .send_event(RunnerEvent::Status(StatusUpdate {
                state: RunnerState::Running,
                message: "hello".into(),
            }))
            .await
            .unwrap();
        let ev = tokio::time::timeout(std::time::Duration::from_millis(100), events_rx.recv())
            .await
            .unwrap()
            .unwrap();
        match ev {
            RunnerEvent::Status(s) => assert_eq!(s.state, RunnerState::Running),
            _ => panic!("expected status"),
        }
    }

    #[tokio::test]
    async fn send_to_unknown_runner_errors() {
        let r = Registry::new();
        let err = r
            .send_command("ghost", ControlCommand::Stop)
            .await
            .unwrap_err();
        matches!(err, RegistryError::NotFound(_));
    }

    #[tokio::test]
    async fn forget_removes_runner() {
        let r = Registry::new();
        let (resp, _) = r.register(req("foo"));
        assert!(r.forget(&resp.runner_id));
        assert!(!r.forget(&resp.runner_id));
        assert!(r.is_empty());
    }

    #[tokio::test]
    async fn runner_for_run_finds_back_reference() {
        let r = Registry::new();
        let (resp, _) = r.register(req("special"));
        assert_eq!(r.runner_for_run("special"), Some(resp.runner_id));
        assert_eq!(r.runner_for_run("nope"), None);
    }
}
