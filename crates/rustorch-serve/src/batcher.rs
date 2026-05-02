//! Dynamic batching task. Owns the inference closure + the per-
//! request mpsc queue.

use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

/// User-supplied inference function. Must be cheap to clone (we
/// store it behind an `Arc`). Synchronous on purpose — the batcher
/// runs it inside `tokio::task::spawn_blocking` so heavy GPU work
/// doesn't block the runtime.
pub type InferFn = Arc<dyn Fn(Vec<Value>) -> Vec<Value> + Send + Sync>;

/// Knobs.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Cap on how many requests get coalesced into a single forward.
    pub max_batch: usize,
    /// Maximum wait time once a request is in the queue. After this,
    /// the batcher fires whatever it has.
    pub max_wait: Duration,
    /// In-flight queue depth. Excess requests get a 503 from the
    /// HTTP layer.
    pub queue_depth: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch: 32,
            max_wait: Duration::from_millis(20),
            queue_depth: 256,
        }
    }
}

/// One queued request — its input plus the channel to send the
/// response back on.
struct Request {
    input: Value,
    reply: oneshot::Sender<Value>,
}

/// Handle the HTTP layer talks to. Cloning is cheap (mpsc::Sender).
#[derive(Clone)]
pub struct Batcher {
    tx: mpsc::Sender<Request>,
    cfg: BatchConfig,
}

impl Batcher {
    /// Spawn the batching task. Returns a handle for callers + the
    /// underlying join handle so tests can wait for shutdown.
    pub fn spawn(infer: InferFn, cfg: BatchConfig) -> Self {
        let (tx, rx) = mpsc::channel(cfg.queue_depth);
        let cfg2 = cfg.clone();
        tokio::spawn(async move {
            run_loop(rx, infer, cfg2).await;
        });
        Self { tx, cfg }
    }

    /// Enqueue an input and await the corresponding output. Returns
    /// `Err` if the queue is full or the batcher has shut down.
    pub async fn predict(&self, input: Value) -> Result<Value, BatcherError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .try_send(Request {
                input,
                reply: reply_tx,
            })
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => BatcherError::Overloaded,
                mpsc::error::TrySendError::Closed(_) => BatcherError::Shutdown,
            })?;
        reply_rx.await.map_err(|_| BatcherError::Shutdown)
    }

    pub fn config(&self) -> &BatchConfig {
        &self.cfg
    }
}

/// Errors observable by the HTTP layer.
#[derive(Debug, thiserror::Error)]
pub enum BatcherError {
    #[error("overloaded — try again later")]
    Overloaded,
    #[error("batcher shut down")]
    Shutdown,
}

async fn run_loop(mut rx: mpsc::Receiver<Request>, infer: InferFn, cfg: BatchConfig) {
    loop {
        // Wait for the first request to arrive — this blocks
        // indefinitely so an idle server consumes no CPU.
        let first = match rx.recv().await {
            Some(r) => r,
            None => return, // sender closed
        };

        let mut batch_inputs = vec![first.input];
        let mut replies = vec![first.reply];

        // Drain up to (max_batch - 1) more requests with a deadline.
        let deadline = tokio::time::Instant::now() + cfg.max_wait;
        while batch_inputs.len() < cfg.max_batch {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            let next = match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Some(req)) => req,
                Ok(None) => return,
                Err(_) => break, // deadline hit
            };
            batch_inputs.push(next.input);
            replies.push(next.reply);
        }

        let infer = infer.clone();
        // Run the model on a blocking thread so the runtime keeps
        // accepting connections.
        let outputs = tokio::task::spawn_blocking(move || infer(batch_inputs))
            .await
            .unwrap_or_else(|e| {
                tracing::error!(error = %e, "infer thread panicked");
                Vec::new()
            });

        // If the model returned a different cardinality, pad with
        // null so each caller still gets *some* response (better
        // than wedging them on the oneshot).
        for (i, reply) in replies.into_iter().enumerate() {
            let out = outputs.get(i).cloned().unwrap_or(Value::Null);
            let _ = reply.send(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn echo() -> InferFn {
        Arc::new(|batch: Vec<Value>| batch)
    }

    #[tokio::test]
    async fn predict_round_trips_single_request() {
        let b = Batcher::spawn(echo(), BatchConfig::default());
        let v = b.predict(json!({"x": 42})).await.unwrap();
        assert_eq!(v["x"], 42);
    }

    #[tokio::test]
    async fn coalesces_concurrent_requests_into_one_batch() {
        let observed_max = Arc::new(AtomicUsize::new(0));
        let max = observed_max.clone();
        let infer: InferFn = Arc::new(move |batch: Vec<Value>| {
            max.fetch_max(batch.len(), Ordering::Relaxed);
            batch
        });

        let b = Batcher::spawn(
            infer,
            BatchConfig {
                max_batch: 8,
                max_wait: Duration::from_millis(20),
                queue_depth: 64,
            },
        );

        let mut futs = Vec::new();
        for i in 0..8 {
            let bb = b.clone();
            futs.push(tokio::spawn(async move {
                bb.predict(json!({"i": i})).await.unwrap()
            }));
        }
        for f in futs {
            let _ = f.await;
        }
        // The infer fn should have seen at least 2 requests in one
        // batch (timing depends on the runtime, hence the >= 2 floor
        // rather than == 8).
        assert!(observed_max.load(Ordering::Relaxed) >= 2);
    }

    #[tokio::test]
    async fn config_is_carried_through() {
        let b = Batcher::spawn(
            echo(),
            BatchConfig {
                max_batch: 7,
                max_wait: Duration::from_millis(13),
                queue_depth: 5,
            },
        );
        assert_eq!(b.config().max_batch, 7);
        assert_eq!(b.config().queue_depth, 5);
    }
}
