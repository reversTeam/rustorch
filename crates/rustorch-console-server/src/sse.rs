//! SSE broadcast hub with monotonic event ids + per-topic 30s ring
//! buffer for `Last-Event-ID` replay.
//!
//! ### Wire shape
//!
//! Every event carries a strictly-increasing `id` that the SSE
//! response surfaces as the standard `id:` field. Browsers then echo
//! the last seen id back via `Last-Event-ID` on reconnect; we use
//! that header to replay anything the client missed within the last
//! 30 seconds before continuing from live broadcast.
//!
//! ### Topology per topic
//!
//! ```text
//!  publish(topic, ...)
//!      │  bumps counter, stamps id, push into ring
//!      ▼
//!  ┌────────────────────────────────┐
//!  │ Topic { ring: VecDeque,        │
//!  │         tx: broadcast::Sender, │
//!  │         next_id: u64 }         │
//!  └─────────────┬──────────────────┘
//!                │ subscribe(last_id?) → Stream
//!                ▼
//!     replay missed events from ring (id > last_id, ts > now-30s)
//!     then forward broadcast::Receiver
//! ```
//!
//! Slow consumers still get the broadcast `Lagged` → mapped to a
//! synthetic `lag` event so the UI can decide to refetch.
//!
//! ### Capacity choices
//!
//! * Ring buffer holds up to 4096 events per topic OR 30 seconds of
//!   history, whichever is hit first. 4096 ≈ 40s at the noisy 100
//!   evt/s limit so the time bound is the active one for hot topics.
//! * Broadcast capacity of 1024 still backs slow consumers — losing
//!   a broadcast slot is recoverable via the ring on the next page
//!   load anyway.

use axum::response::sse::{Event, KeepAlive, Sse};
use futures::Stream;
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

const BROADCAST_CAPACITY: usize = 1024;
const RING_CAPACITY: usize = 4096;
const RING_RETENTION: Duration = Duration::from_secs(30);

/// One event published on a topic. The `id` is strictly increasing
/// per-topic (counter starts at 1) and lets reconnecting clients ask
/// for "everything since N".
#[derive(Debug, Clone)]
pub struct HubEvent {
    pub id: u64,
    pub kind: &'static str,
    pub payload: serde_json::Value,
}

/// Per-topic state — broadcast sender for live fan-out + ring buffer
/// for short-window replay. Wrapped in an `Arc` so subscribers can
/// hold a reference without keeping the registry mutex.
#[derive(Debug)]
struct Topic {
    tx: broadcast::Sender<HubEvent>,
    next_id: u64,
    ring: VecDeque<(Instant, HubEvent)>,
}

impl Topic {
    fn new() -> Self {
        let (tx, _) = broadcast::channel(BROADCAST_CAPACITY);
        Self {
            tx,
            next_id: 1,
            ring: VecDeque::new(),
        }
    }

    /// Drop entries older than `RING_RETENTION` from the head. Called
    /// on every publish + subscribe so the ring stays bounded even
    /// on bursty topics.
    fn evict_old(&mut self) {
        let cutoff = Instant::now() - RING_RETENTION;
        while let Some((ts, _)) = self.ring.front() {
            if *ts < cutoff {
                self.ring.pop_front();
            } else {
                break;
            }
        }
        while self.ring.len() > RING_CAPACITY {
            self.ring.pop_front();
        }
    }
}

/// Cloneable handle to the hub. Internally `Arc<Mutex<HashMap>>` so
/// publishers + subscribers can share it across tasks. The mutex is
/// short-held (no `.await` under it).
#[derive(Clone, Default)]
pub struct Hub {
    inner: Arc<Mutex<HashMap<String, Topic>>>,
}

/// Result of a subscribe-with-replay call. The replay events are
/// drained eagerly so the caller can flush them onto the SSE stream
/// before hooking up the live receiver.
pub struct Subscription {
    pub replay: Vec<HubEvent>,
    pub rx: broadcast::Receiver<HubEvent>,
}

impl Hub {
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe to `topic` and (optionally) replay every buffered
    /// event with `id > last_id`. The first call for a given topic
    /// lazily creates the underlying topic state.
    pub fn subscribe(&self, topic: &str, last_id: Option<u64>) -> Subscription {
        let mut map = self.inner.lock().unwrap();
        let t = map.entry(topic.to_string()).or_insert_with(Topic::new);
        t.evict_old();

        let replay: Vec<HubEvent> = match last_id {
            Some(after) => t
                .ring
                .iter()
                .filter(|(_, ev)| ev.id > after)
                .map(|(_, ev)| ev.clone())
                .collect(),
            None => Vec::new(),
        };
        Subscription {
            replay,
            rx: t.tx.subscribe(),
        }
    }

    /// Publish `value` on `topic`. Returns the event id assigned.
    /// Silently drops if serialization fails. Stamps the event into
    /// both the live broadcast and the per-topic ring buffer.
    pub fn publish<T: Serialize>(&self, topic: &str, kind: &'static str, value: &T) -> u64 {
        let payload = match serde_json::to_value(value) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, topic, "hub: drop unserializable payload");
                return 0;
            },
        };

        let mut map = self.inner.lock().unwrap();
        let t = map.entry(topic.to_string()).or_insert_with(Topic::new);
        let id = t.next_id;
        t.next_id += 1;

        let ev = HubEvent { id, kind, payload };
        t.ring.push_back((Instant::now(), ev.clone()));
        t.evict_old();
        // Best-effort send. If there are zero subscribers, broadcast
        // returns Err — that's fine, the ring still has it.
        let _ = t.tx.send(ev);
        id
    }

    /// Number of currently active subscribers. Used by tests + a
    /// future `/debug/sse` endpoint.
    pub fn subscriber_count(&self, topic: &str) -> usize {
        let map = self.inner.lock().unwrap();
        map.get(topic).map(|t| t.tx.receiver_count()).unwrap_or(0)
    }

    /// How many events are currently held in the replay ring. Tests
    /// use this to assert eviction.
    pub fn ring_len(&self, topic: &str) -> usize {
        let map = self.inner.lock().unwrap();
        map.get(topic).map(|t| t.ring.len()).unwrap_or(0)
    }
}

/// Convert a `Subscription` into an axum `Sse` response. Replay
/// events are emitted first (in id order), then live broadcast
/// events. `Lagged` is mapped to a synthetic `lag` event so a slow
/// consumer can refetch state instead of being silently dropped.
pub fn sse_response(sub: Subscription) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let replay_stream = futures::stream::iter(
        sub.replay
            .into_iter()
            .map(Ok::<_, broadcast::error::RecvError>),
    );
    let live_stream = BroadcastStream::new(sub.rx).map(|res| {
        res.map_err(|e| match e {
            tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(_) => {
                broadcast::error::RecvError::Lagged(0)
            },
        })
    });

    let combined = replay_stream.chain(live_stream).map(|res| {
        let event = match res {
            Ok(ev) => Event::default()
                .id(ev.id.to_string())
                .event(ev.kind)
                .json_data(ev.payload)
                .unwrap_or_else(|_| Event::default().data("{}")),
            Err(_lag) => Event::default().event("lag").data("{}"),
        };
        Ok::<_, Infallible>(event)
    });
    Sse::new(combined).keep_alive(KeepAlive::default())
}

/// Convenience: subscribe + wrap into an `Sse` response in one shot.
pub fn sse_for(
    hub: &Hub,
    topic: &str,
    last_id: Option<u64>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    sse_response(hub.subscribe(topic, last_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn publish_assigns_monotonic_ids() {
        let hub = Hub::new();
        let id1 = hub.publish("t", "evt", &serde_json::json!({"v": 1}));
        let id2 = hub.publish("t", "evt", &serde_json::json!({"v": 2}));
        let id3 = hub.publish("t", "evt", &serde_json::json!({"v": 3}));
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    #[tokio::test]
    async fn subscribe_with_no_last_id_skips_replay() {
        let hub = Hub::new();
        hub.publish("t", "evt", &serde_json::json!({"v": 1}));
        let sub = hub.subscribe("t", None);
        assert!(sub.replay.is_empty());
    }

    #[tokio::test]
    async fn subscribe_with_last_id_replays_missed_events() {
        let hub = Hub::new();
        hub.publish("t", "evt", &serde_json::json!({"v": 1}));
        hub.publish("t", "evt", &serde_json::json!({"v": 2}));
        hub.publish("t", "evt", &serde_json::json!({"v": 3}));

        // Client claims it last saw id=1 → expect replay of id=2 and id=3.
        let sub = hub.subscribe("t", Some(1));
        let ids: Vec<u64> = sub.replay.iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![2, 3]);
    }

    #[tokio::test]
    async fn subscribe_with_last_id_higher_than_max_returns_empty() {
        let hub = Hub::new();
        hub.publish("t", "evt", &serde_json::json!({"v": 1}));
        let sub = hub.subscribe("t", Some(999));
        assert!(sub.replay.is_empty());
    }

    #[tokio::test]
    async fn live_events_arrive_after_replay() {
        let hub = Hub::new();
        hub.publish("t", "evt", &serde_json::json!({"v": 1}));
        let mut sub = hub.subscribe("t", Some(0));
        // Replay should contain id=1.
        assert_eq!(sub.replay.len(), 1);
        // Now publish a live event and read it via the receiver.
        hub.publish("t", "evt", &serde_json::json!({"v": 2}));
        let live = timeout(Duration::from_millis(100), sub.rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(live.id, 2);
    }

    #[tokio::test]
    async fn ring_caps_at_4096_entries() {
        let hub = Hub::new();
        // Push enough events to overflow the 4096 cap.
        for i in 0..5000 {
            hub.publish("t", "evt", &serde_json::json!({"v": i}));
        }
        // Eviction by count keeps at most RING_CAPACITY entries.
        assert!(hub.ring_len("t") <= RING_CAPACITY);
    }

    #[tokio::test]
    async fn no_subscribers_drops_silently_but_buffers() {
        let hub = Hub::new();
        // Should not panic even though nobody is subscribed yet.
        hub.publish("nobody.listening", "ping", &serde_json::json!({}));
        assert_eq!(hub.subscriber_count("nobody.listening"), 0);
        // Late subscriber with last_id=0 still gets the buffered event.
        let sub = hub.subscribe("nobody.listening", Some(0));
        assert_eq!(sub.replay.len(), 1);
    }

    #[tokio::test]
    async fn multiple_subscribers_each_get_live_events() {
        let hub = Hub::new();
        let mut a = hub.subscribe("t", None);
        let mut b = hub.subscribe("t", None);
        hub.publish("t", "evt", &serde_json::json!({"v": 1}));
        let ea = timeout(Duration::from_millis(100), a.rx.recv())
            .await
            .unwrap()
            .unwrap();
        let eb = timeout(Duration::from_millis(100), b.rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(ea.payload["v"], 1);
        assert_eq!(eb.payload["v"], 1);
        assert_eq!(ea.id, eb.id);
    }
}
