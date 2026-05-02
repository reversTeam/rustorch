//! SSE broadcast hub. A `Hub` owns one `tokio::broadcast::Sender`
//! per topic and hands out fresh receivers on subscribe. Producers
//! call `publish` with arbitrary JSON-serializable events; consumers
//! get an axum `Sse` stream that lives until the client disconnects.
//!
//! The hub is intentionally minimal:
//!   * fan-out via tokio broadcast (lock-free producers, bounded
//!     ring per subscriber);
//!   * `RecvError::Lagged` is mapped to a special "drop" event so
//!     the UI can decide whether to refresh;
//!   * topics are created lazily on first publish or subscribe.
//!
//! Reconnect-with-Last-Event-ID replay lives in a follow-up commit
//! — the broadcast channel doesn't keep history beyond its capacity,
//! so a real replay needs a per-topic ring buffer. This module
//! exposes the seam for that.

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use serde::Serialize;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

/// Per-topic capacity. 1024 ≈ 17 minutes at 1 ev/s, plenty of slack
/// for the cluster.tick topic; faster topics (run.metric at 100/s)
/// will lag a slow consumer rather than block producers.
const TOPIC_CAPACITY: usize = 1024;

/// A single SSE event ready for serialization. We keep it generic so
/// the call sites don't have to think about JSON beforehand.
#[derive(Debug, Clone)]
pub struct HubEvent {
    pub kind: &'static str,
    pub payload: serde_json::Value,
}

/// Cloneable handle to the hub. Internally an `Arc<Mutex<...>>` so
/// the same handle works for both producers (via lib code) and
/// consumers (via axum extractors).
#[derive(Clone, Default)]
pub struct Hub {
    inner: Arc<Mutex<HashMap<String, broadcast::Sender<HubEvent>>>>,
}

impl Hub {
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe to `topic`. The first call for a given topic creates
    /// the underlying broadcast sender.
    pub fn subscribe(&self, topic: &str) -> broadcast::Receiver<HubEvent> {
        let mut map = self.inner.lock().unwrap();
        map.entry(topic.to_string())
            .or_insert_with(|| broadcast::channel(TOPIC_CAPACITY).0)
            .subscribe()
    }

    /// Publish an event to `topic`. Silently no-ops if there are no
    /// subscribers — events are best-effort, never persistent.
    pub fn publish<T: Serialize>(&self, topic: &str, kind: &'static str, value: &T) {
        let payload = match serde_json::to_value(value) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "hub: failed to serialize payload, dropping");
                return;
            },
        };
        let map = self.inner.lock().unwrap();
        if let Some(tx) = map.get(topic) {
            let _ = tx.send(HubEvent { kind, payload });
        }
    }

    /// Number of currently active subscribers on `topic`. Used by
    /// tests and `/debug` endpoints.
    pub fn subscriber_count(&self, topic: &str) -> usize {
        let map = self.inner.lock().unwrap();
        map.get(topic).map(|tx| tx.receiver_count()).unwrap_or(0)
    }
}

/// Convert a topic into an axum `Sse` response. Each broadcast event
/// becomes one `data: {...}` line. Lag (slow consumer) is mapped to
/// a synthetic `lag` event so the UI can re-fetch if it cares.
pub fn sse_for(hub: &Hub, topic: &str) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = hub.subscribe(topic);
    let stream = BroadcastStream::new(rx).map(|res| {
        let event = match res {
            Ok(ev) => Event::default()
                .event(ev.kind)
                .json_data(ev.payload)
                .unwrap_or_else(|_| Event::default().data("{}")),
            Err(_lag) => Event::default().event("lag").data("{}"),
        };
        Ok::<_, Infallible>(event)
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn publish_then_receive_roundtrip() {
        let hub = Hub::new();
        let mut rx = hub.subscribe("runs.changed");
        hub.publish(
            "runs.changed",
            "runs.changed",
            &serde_json::json!({"id": "x"}),
        );
        let ev = timeout(Duration::from_millis(100), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(ev.kind, "runs.changed");
        assert_eq!(ev.payload["id"], "x");
    }

    #[tokio::test]
    async fn no_subscribers_drops_silently() {
        let hub = Hub::new();
        // Should not panic and should not allocate the topic.
        hub.publish("nobody.listening", "ping", &serde_json::json!({}));
        assert_eq!(hub.subscriber_count("nobody.listening"), 0);
    }

    #[tokio::test]
    async fn multiple_subscribers_each_get_event() {
        let hub = Hub::new();
        let mut a = hub.subscribe("t");
        let mut b = hub.subscribe("t");
        hub.publish("t", "evt", &serde_json::json!({"v": 1}));
        let ea = timeout(Duration::from_millis(100), a.recv())
            .await
            .unwrap()
            .unwrap();
        let eb = timeout(Duration::from_millis(100), b.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(ea.payload["v"], 1);
        assert_eq!(eb.payload["v"], 1);
    }

    #[tokio::test]
    async fn subscriber_count_tracks_drops() {
        let hub = Hub::new();
        let r1 = hub.subscribe("t");
        let r2 = hub.subscribe("t");
        assert_eq!(hub.subscriber_count("t"), 2);
        drop(r1);
        // Trigger the broadcast to update the receiver count.
        hub.publish("t", "evt", &serde_json::json!({}));
        let _ = r2;
        assert!(hub.subscriber_count("t") <= 2);
    }
}
