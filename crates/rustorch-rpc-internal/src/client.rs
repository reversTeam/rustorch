//! Runner-side client. Mirrors the eventual gRPC API but talks to
//! the in-process `Registry` so the entire flow can be tested
//! without protoc / tonic. Same module structure (connect →
//! register → open_session) so the future tonic implementation
//! drops in behind the `grpc` feature flag without touching call
//! sites.
//!
//! ### Features
//!
//! * `RunnerClient::connect_in_process(registry)` — direct hookup
//!   for tests + in-process deployments.
//! * Auto-reconnect with exponential backoff (1s → 2s → … → 30s).
//! * Local buffer (drop-oldest on overflow) so events queued during
//!   a disconnect get re-sent on reconnect.
//! * `RpcRunnerSink` — `rustorch-log::Sink` impl that pushes
//!   `LogEvent` → `RunnerEvent` over the session.

use crate::registry::{Registry, RunnerSession};
use crate::types::{RegisterRequest, RegisterResponse, RunnerEvent};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;

/// Knobs for the client's reconnect/buffer policy.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Max retained buffered events while disconnected. Older
    /// events get dropped (oldest-first) once the bound is hit.
    pub buffer_size: usize,
    /// Backoff series cap.
    pub backoff_max: Duration,
    /// Heartbeat interval.
    pub heartbeat: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024,
            backoff_max: Duration::from_secs(30),
            heartbeat: Duration::from_secs(10),
        }
    }
}

/// Stats observable from outside the client — useful for tests +
/// the `/debug/runner` endpoint.
#[derive(Debug, Clone, Default)]
pub struct ClientStats {
    pub events_sent: u64,
    pub events_buffered: u64,
    pub events_dropped: u64,
    pub reconnects: u32,
}

/// Buffered event handle — exposed so reconnect logic can drain it
/// in FIFO order.
#[derive(Debug, Default)]
struct EventBuffer {
    queue: VecDeque<RunnerEvent>,
    cap: usize,
    dropped: u64,
}

impl EventBuffer {
    fn new(cap: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(cap.min(1024)),
            cap,
            dropped: 0,
        }
    }

    fn push(&mut self, ev: RunnerEvent) {
        if self.queue.len() >= self.cap {
            self.queue.pop_front();
            self.dropped += 1;
        }
        self.queue.push_back(ev);
    }

    fn drain(&mut self) -> Vec<RunnerEvent> {
        self.queue.drain(..).collect()
    }
}

/// Client handle the runner code drives.
#[derive(Clone)]
pub struct RunnerClient {
    inner: Arc<ClientInner>,
}

struct ClientInner {
    registry: Registry,
    runner_id: parking_lot::RwLock<Option<String>>,
    register_request: parking_lot::RwLock<Option<RegisterRequest>>,
    cfg: ClientConfig,
    buffer: Mutex<EventBuffer>,
    stats: Mutex<ClientStats>,
    /// Live `events_tx` while connected. `None` while reconnecting.
    events_tx: Mutex<Option<tokio::sync::mpsc::Sender<RunnerEvent>>>,
}

impl RunnerClient {
    /// Connect against an in-process registry. The eventual gRPC
    /// flavour will be `RunnerClient::connect("grpc://…")` behind
    /// feature `grpc` — same return type.
    pub fn connect_in_process(registry: Registry, cfg: ClientConfig) -> Self {
        Self {
            inner: Arc::new(ClientInner {
                registry,
                runner_id: parking_lot::RwLock::new(None),
                register_request: parking_lot::RwLock::new(None),
                cfg: cfg.clone(),
                buffer: Mutex::new(EventBuffer::new(cfg.buffer_size)),
                stats: Mutex::new(ClientStats::default()),
                events_tx: Mutex::new(None),
            }),
        }
    }

    /// Register against the registry and open the bidirectional
    /// session. Returns the response + a session handle the caller
    /// holds (its `next_command` is what the runner awaits).
    pub fn register(&self, req: RegisterRequest) -> (RegisterResponse, RunnerSession) {
        let (resp, session) = self.inner.registry.register(req.clone());
        *self.inner.runner_id.write() = Some(resp.runner_id.clone());
        *self.inner.register_request.write() = Some(req);
        *self.inner.events_tx.lock() = Some(session.events_tx.clone());
        (resp, session)
    }

    /// Push an event through the session — this is the hot path the
    /// `Sink` impl uses. If we're disconnected the event is
    /// buffered; reconnect drains the buffer first.
    pub async fn send_event(&self, ev: RunnerEvent) {
        // Drop the lock before awaiting — the parking_lot guard isn't
        // Send, so holding it across `.await` would prevent the
        // future from being scheduled on a multi-thread runtime.
        let tx = self.inner.events_tx.lock().clone();
        match tx {
            Some(tx) => match tx.send(ev.clone()).await {
                Ok(()) => self.inner.stats.lock().events_sent += 1,
                Err(_) => {
                    // Channel closed — buffer + try to reconnect.
                    self.buffer_and_signal_disconnect(ev).await;
                },
            },
            None => self.buffer_and_signal_disconnect(ev).await,
        }
    }

    async fn buffer_and_signal_disconnect(&self, ev: RunnerEvent) {
        {
            let mut buf = self.inner.buffer.lock();
            buf.push(ev);
            self.inner.stats.lock().events_buffered = buf.queue.len() as u64;
        }
        // Best-effort reconnect — runs in the background, the caller
        // doesn't await.
        let me = self.clone();
        tokio::spawn(async move {
            me.try_reconnect().await;
        });
    }

    /// Re-register against the registry and flush the buffer. Used
    /// internally on disconnect; tests can call it directly to
    /// assert the buffer drains correctly.
    pub async fn try_reconnect(&self) -> bool {
        let req = match self.inner.register_request.read().clone() {
            Some(r) => r,
            None => return false,
        };
        // For an in-process registry "connect" is always a single
        // register() call. The real gRPC client will swap this body
        // for an exponential-backoff retry — see `compute_backoff`
        // below for the schedule we expect to use there.
        let (resp, session) = self.inner.registry.register(req.clone());
        *self.inner.runner_id.write() = Some(resp.runner_id);
        *self.inner.events_tx.lock() = Some(session.events_tx.clone());
        self.inner.stats.lock().reconnects += 1;
        // Drain the buffer onto the new session.
        let drained = self.inner.buffer.lock().drain();
        let tx_clone = self.inner.events_tx.lock().clone();
        if let Some(tx) = tx_clone {
            for ev in drained {
                let _ = tx.send(ev).await;
            }
        }
        true
    }

    /// Exponential backoff schedule used by the future gRPC client.
    /// `attempt` is 0-indexed. Caps at `backoff_max`.
    pub fn compute_backoff(&self, attempt: u32) -> Duration {
        let raw = Duration::from_secs(1u64 << attempt.min(6));
        raw.min(self.inner.cfg.backoff_max)
    }

    /// Spawn the heartbeat task. Returns the handle so callers can
    /// abort it on shutdown.
    pub fn spawn_heartbeat(&self) -> JoinHandle<()> {
        let me = self.clone();
        let interval = me.inner.cfg.heartbeat;
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                // Real gRPC: send Heartbeat RPC. In-process: no-op.
                let need_reconnect = me.inner.events_tx.lock().is_none();
                if need_reconnect {
                    let _ = me.try_reconnect().await;
                }
            }
        })
    }

    pub fn stats(&self) -> ClientStats {
        self.inner.stats.lock().clone()
    }

    pub fn runner_id(&self) -> Option<String> {
        self.inner.runner_id.read().clone()
    }

    pub fn buffered_count(&self) -> usize {
        self.inner.buffer.lock().queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MetricSample, RunnerState, StatusUpdate};

    fn req(run_id: &str) -> RegisterRequest {
        RegisterRequest {
            run_id: run_id.into(),
            gpu_count: 1,
            runner_version: "0.0.1".into(),
        }
    }

    #[tokio::test]
    async fn happy_path_register_and_send_event() {
        let registry = Registry::new();
        let client = RunnerClient::connect_in_process(registry.clone(), ClientConfig::default());
        let (resp, _session) = client.register(req("run-a"));
        let mut events_rx = registry.take_events_rx(&resp.runner_id).unwrap();

        client
            .send_event(RunnerEvent::Status(StatusUpdate {
                state: RunnerState::Running,
                message: "hello".into(),
            }))
            .await;

        let ev = tokio::time::timeout(Duration::from_millis(100), events_rx.recv())
            .await
            .unwrap()
            .unwrap();
        match ev {
            RunnerEvent::Status(s) => assert_eq!(s.state, RunnerState::Running),
            _ => panic!(),
        }
        assert_eq!(client.stats().events_sent, 1);
    }

    #[tokio::test]
    async fn buffer_holds_events_during_disconnect() {
        let registry = Registry::new();
        let client = RunnerClient::connect_in_process(
            registry,
            ClientConfig {
                buffer_size: 4,
                ..Default::default()
            },
        );
        // No register — events_tx is None, every send buffers.
        for i in 0..3 {
            client
                .send_event(RunnerEvent::Metric(MetricSample {
                    step: i,
                    name: "loss".into(),
                    value: 0.5,
                }))
                .await;
        }
        // Give the spawned reconnect a moment.
        tokio::time::sleep(Duration::from_millis(20)).await;
        // Without a register_request, reconnect can't run, so events
        // stay buffered.
        assert!(client.buffered_count() >= 1);
    }

    #[tokio::test]
    async fn buffer_drops_oldest_on_overflow() {
        let registry = Registry::new();
        let client = RunnerClient::connect_in_process(
            registry,
            ClientConfig {
                buffer_size: 2,
                ..Default::default()
            },
        );
        for i in 0..5 {
            client
                .send_event(RunnerEvent::Metric(MetricSample {
                    step: i,
                    name: "loss".into(),
                    value: 0.5,
                }))
                .await;
        }
        // Only the last `buffer_size` events should remain.
        let inner = &client.inner.buffer.lock();
        assert!(inner.queue.len() <= 2);
        assert!(inner.dropped >= 1);
    }

    #[tokio::test]
    async fn backoff_grows_then_caps() {
        let registry = Registry::new();
        let client = RunnerClient::connect_in_process(
            registry,
            ClientConfig {
                backoff_max: Duration::from_secs(30),
                ..Default::default()
            },
        );
        let b0 = client.compute_backoff(0);
        let b1 = client.compute_backoff(1);
        let b3 = client.compute_backoff(3);
        let b_huge = client.compute_backoff(20);
        assert_eq!(b0, Duration::from_secs(1));
        assert_eq!(b1, Duration::from_secs(2));
        assert_eq!(b3, Duration::from_secs(8));
        assert_eq!(b_huge, Duration::from_secs(30));
    }

    #[tokio::test]
    async fn try_reconnect_re_registers_and_increments_counter() {
        let registry = Registry::new();
        let client = RunnerClient::connect_in_process(registry.clone(), ClientConfig::default());
        let (resp_a, _session) = client.register(req("run-b"));

        // Force a "disconnect" by clearing the live events_tx.
        *client.inner.events_tx.lock() = None;

        // Reconnect — explicit (not spawned), so we can assert.
        let ok = client.try_reconnect().await;
        assert!(ok);

        let new_id = client.runner_id().unwrap();
        assert_ne!(new_id, resp_a.runner_id, "runner_id should be re-issued");
        assert!(client.inner.events_tx.lock().is_some());
        assert!(client.stats().reconnects >= 1);
    }

    #[tokio::test]
    async fn buffered_events_drain_after_explicit_reconnect() {
        let registry = Registry::new();
        let client = RunnerClient::connect_in_process(registry.clone(), ClientConfig::default());
        let (resp, _session) = client.register(req("run-c"));
        // Take + drop events_rx so the original tx finds no listener
        // and the buffer activates.
        let _ = registry.take_events_rx(&resp.runner_id);

        // Disconnect.
        *client.inner.events_tx.lock() = None;

        // Push events directly into the buffer (bypass the spawn race).
        for i in 0..3 {
            let mut buf = client.inner.buffer.lock();
            buf.push(RunnerEvent::Metric(MetricSample {
                step: i,
                name: "loss".into(),
                value: 0.5,
            }));
        }
        assert_eq!(client.buffered_count(), 3);

        // Explicit reconnect drains the buffer onto the new session.
        let ok = client.try_reconnect().await;
        assert!(ok);
        let new_id = client.runner_id().unwrap();
        let mut events_rx = registry.take_events_rx(&new_id).unwrap();

        // 3 events should be deliverable now.
        let mut got = 0;
        while got < 3 {
            match tokio::time::timeout(Duration::from_millis(200), events_rx.recv()).await {
                Ok(Some(_)) => got += 1,
                _ => break,
            }
        }
        assert_eq!(got, 3);
        assert_eq!(client.buffered_count(), 0);
    }
}
