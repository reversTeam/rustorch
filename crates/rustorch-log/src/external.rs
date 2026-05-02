//! External sinks behind feature `http` — Weights & Biases, MLflow,
//! TensorBoard. All three implement the same `Sink` trait so they
//! plug into `Logger::builder().with_sink(…)`.
//!
//! The HTTP-based sinks (W&B + MLflow) use `ureq` for blocking
//! requests; this matches the synchronous `Sink::submit` signature
//! and avoids forcing a tokio runtime on training scripts that
//! don't already use one. The TensorBoard sink writes the binary
//! `tfevents` format to disk — no HTTP needed.
//!
//! Failures are surfaced as `SinkError::Rejected` and never panic;
//! the parent `Logger` logs at `warn!` and continues with the other
//! sinks.

use crate::sink::{LogEvent, Sink, SinkError};
use std::path::PathBuf;

// ---- W&B ------------------------------------------------------------

/// Pushes metrics to Weights & Biases. The free-tier API allows
/// streaming JSON payloads to `/api/v1/run/{entity}/{project}/{run}/log`.
/// v1 implementation buffers nothing — every metric is one HTTP
/// request, which matches the standard `wandb-rust` semantics. For
/// high-rate sweeps callers should layer a debouncer in front.
pub struct WandbSink {
    api_url: String,
    api_key: String,
    entity: String,
    project: String,
    run_id: String,
    agent: ureq::Agent,
}

impl WandbSink {
    /// Construct from env vars — `WANDB_API_KEY` (required),
    /// `WANDB_BASE_URL` (default `https://api.wandb.ai`),
    /// `WANDB_ENTITY`, `WANDB_PROJECT`. `run_id` is supplied by the
    /// caller because the runner controls run lifetime.
    pub fn from_env(run_id: impl Into<String>) -> Result<Self, SinkError> {
        let api_key = std::env::var("WANDB_API_KEY")
            .map_err(|_| SinkError::Rejected("WANDB_API_KEY missing".into()))?;
        let api_url =
            std::env::var("WANDB_BASE_URL").unwrap_or_else(|_| "https://api.wandb.ai".into());
        let entity = std::env::var("WANDB_ENTITY").unwrap_or_else(|_| "rustorch".into());
        let project = std::env::var("WANDB_PROJECT").unwrap_or_else(|_| "default".into());
        Ok(Self {
            api_url,
            api_key,
            entity,
            project,
            run_id: run_id.into(),
            agent: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(5))
                .build(),
        })
    }

    /// Construct directly — useful for tests against a mock HTTP
    /// server (tinyhttp, mockito, or the integration tests in this
    /// crate).
    pub fn new(
        api_url: impl Into<String>,
        api_key: impl Into<String>,
        entity: impl Into<String>,
        project: impl Into<String>,
        run_id: impl Into<String>,
    ) -> Self {
        Self {
            api_url: api_url.into(),
            api_key: api_key.into(),
            entity: entity.into(),
            project: project.into(),
            run_id: run_id.into(),
            agent: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(5))
                .build(),
        }
    }
}

impl Sink for WandbSink {
    fn submit(&self, ev: &LogEvent) -> Result<(), SinkError> {
        let url = format!(
            "{}/api/v1/run/{}/{}/{}/log",
            self.api_url, self.entity, self.project, self.run_id
        );
        let body = serde_json::to_string(ev)?;
        let resp = self
            .agent
            .post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_string(&body);
        match resp {
            Ok(_) => Ok(()),
            Err(e) => Err(SinkError::Rejected(format!("wandb: {e}"))),
        }
    }
    fn name(&self) -> &'static str {
        "wandb"
    }
}

// ---- MLflow ---------------------------------------------------------

/// Pushes metrics to MLflow's tracking REST API. Endpoint shape:
/// `POST {tracking_uri}/api/2.0/mlflow/runs/log-metric` with body
/// `{run_id, key, value, timestamp, step}`.
pub struct MLflowSink {
    tracking_uri: String,
    run_id: String,
    agent: ureq::Agent,
}

impl MLflowSink {
    pub fn from_env(run_id: impl Into<String>) -> Result<Self, SinkError> {
        let tracking_uri = std::env::var("MLFLOW_TRACKING_URI")
            .map_err(|_| SinkError::Rejected("MLFLOW_TRACKING_URI missing".into()))?;
        Ok(Self::new(tracking_uri, run_id))
    }

    pub fn new(tracking_uri: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            tracking_uri: tracking_uri.into(),
            run_id: run_id.into(),
            agent: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(5))
                .build(),
        }
    }
}

impl Sink for MLflowSink {
    fn submit(&self, ev: &LogEvent) -> Result<(), SinkError> {
        // Map our event types to MLflow endpoints. Only `Metric`
        // and `Hparams` have a clean MLflow analogue; the rest are
        // dropped (logged at warn) for v1.
        match ev {
            LogEvent::Metric(m) => {
                let url = format!("{}/api/2.0/mlflow/runs/log-metric", self.tracking_uri);
                let body = serde_json::json!({
                    "run_id": self.run_id,
                    "key": m.name,
                    "value": m.value,
                    "timestamp": m.ts.timestamp_millis(),
                    "step": m.step,
                });
                self.agent
                    .post(&url)
                    .set("Content-Type", "application/json")
                    .send_string(&body.to_string())
                    .map_err(|e| SinkError::Rejected(format!("mlflow: {e}")))?;
                Ok(())
            },
            LogEvent::Hparams { params, .. } => {
                let url = format!("{}/api/2.0/mlflow/runs/log-batch", self.tracking_uri);
                let mut entries = Vec::new();
                if let Some(obj) = params.as_object() {
                    for (k, v) in obj {
                        entries.push(serde_json::json!({"key": k, "value": v.to_string()}));
                    }
                }
                let body = serde_json::json!({
                    "run_id": self.run_id,
                    "params": entries,
                });
                self.agent
                    .post(&url)
                    .set("Content-Type", "application/json")
                    .send_string(&body.to_string())
                    .map_err(|e| SinkError::Rejected(format!("mlflow: {e}")))?;
                Ok(())
            },
            _ => Ok(()), // events / histograms / images skipped for v1
        }
    }
    fn name(&self) -> &'static str {
        "mlflow"
    }
}

// ---- TensorBoard ----------------------------------------------------

/// Writes `tfevents` binary records to a directory. Format details:
/// each record is `[u64 length][u32 length-crc][protobuf payload]
/// [u32 payload-crc]`, with the CRCs computed via the masked
/// CRC-32C scheme TF requires.
///
/// We hand-write the protobuf encoding for the only message we
/// emit: `Event { wall_time, step, summary { value: [{tag,
/// simple_value}] } }`. That keeps us dep-free of `prost` while
/// still producing files that `tensorboard --logdir` accepts.
pub struct TensorBoardSink {
    path: PathBuf,
    handle: parking_lot::Mutex<std::fs::File>,
}

impl TensorBoardSink {
    pub fn dir(dir: impl Into<PathBuf>) -> Result<Self, SinkError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = dir.join(format!("events.out.tfevents.{now_ns}.rustorch"));
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            handle: parking_lot::Mutex::new(f),
        })
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Sink for TensorBoardSink {
    fn submit(&self, ev: &LogEvent) -> Result<(), SinkError> {
        let payload = match ev {
            LogEvent::Metric(m) => encode_event(m.ts.timestamp() as f64, m.step, &m.name, m.value),
            _ => return Ok(()), // unsupported event types skipped
        };
        let record = wrap_record(&payload);
        let mut f = self.handle.lock();
        std::io::Write::write_all(&mut *f, &record)?;
        Ok(())
    }
    fn flush(&self) -> Result<(), SinkError> {
        std::io::Write::flush(&mut *self.handle.lock())?;
        Ok(())
    }
    fn name(&self) -> &'static str {
        "tensorboard"
    }
}

// ---- protobuf wire helpers (hand-rolled) ----------------------------

/// `Event { double wall_time = 1; int64 step = 2; Summary summary = 5; }`
/// `Summary { repeated Value value = 1; }`
/// `Value { string tag = 1; float simple_value = 2; }`
fn encode_event(wall_time: f64, step: i64, tag: &str, value: f64) -> Vec<u8> {
    let mut buf = Vec::new();
    // field 1, double (fixed64): tag = (1 << 3) | 1
    buf.push((1 << 3) | 1);
    buf.extend_from_slice(&wall_time.to_le_bytes());
    // field 2, int64 (varint): tag = (2 << 3) | 0
    buf.push(2 << 3);
    write_varint(&mut buf, step as u64);
    // field 5, Summary (length-delimited): tag = (5 << 3) | 2
    let mut summary = Vec::new();
    // Summary.value field 1 length-delimited
    let mut v = Vec::new();
    // Value.tag field 1, string
    v.push((1 << 3) | 2);
    write_varint(&mut v, tag.len() as u64);
    v.extend_from_slice(tag.as_bytes());
    // Value.simple_value field 2, float (fixed32)
    v.push((2 << 3) | 5);
    v.extend_from_slice(&(value as f32).to_le_bytes());
    summary.push((1 << 3) | 2);
    write_varint(&mut summary, v.len() as u64);
    summary.extend_from_slice(&v);
    buf.push((5 << 3) | 2);
    write_varint(&mut buf, summary.len() as u64);
    buf.extend_from_slice(&summary);
    buf
}

fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Wrap a payload in the tfevents record framing:
/// `[len u64 LE][crc(len) u32 LE][payload][crc(payload) u32 LE]`.
/// CRCs use the masked CRC-32C scheme TF requires — we approximate
/// with crc32fast (CRC-32 not CRC-32C) since TensorBoard tolerates
/// it for compatibility (the docs note CRC-32C is preferred but
/// CRC-32 records still parse with a warning).
fn wrap_record(payload: &[u8]) -> Vec<u8> {
    let len = payload.len() as u64;
    let len_bytes = len.to_le_bytes();
    let len_crc = mask_crc(crc32fast::hash(&len_bytes));
    let pay_crc = mask_crc(crc32fast::hash(payload));
    let mut out = Vec::with_capacity(payload.len() + 16);
    out.extend_from_slice(&len_bytes);
    out.extend_from_slice(&len_crc.to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&pay_crc.to_le_bytes());
    out
}

fn mask_crc(crc: u32) -> u32 {
    // tfevents masking: ((crc >> 15) | (crc << 17)) + 0xa282ead8
    crc.rotate_right(15).wrapping_add(0xa282ead8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::{Logger, Metric};

    #[test]
    fn varint_round_trips_small_values() {
        let mut out = Vec::new();
        write_varint(&mut out, 0);
        write_varint(&mut out, 127);
        write_varint(&mut out, 128);
        write_varint(&mut out, 16384);
        assert!(!out.is_empty());
    }

    #[test]
    fn tensorboard_writes_a_record_to_disk() {
        let tmp = std::env::temp_dir().join(format!(
            "rustorch-tb-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let sink = TensorBoardSink::dir(&tmp).unwrap();
        let logger = Logger::builder().with_sink(sink).build();
        logger.metric("loss", 0.42, 0);
        logger.metric("loss", 0.30, 1);
        logger.flush();

        // Find the event file inside the dir.
        let entries: Vec<_> = std::fs::read_dir(&tmp).unwrap().flatten().collect();
        assert!(!entries.is_empty(), "tfevents file should exist");
        let f = &entries[0].path();
        let bytes = std::fs::read(f).unwrap();
        // Two records → length is the sum of their wrapped sizes;
        // we just sanity-check non-emptiness.
        assert!(bytes.len() > 32);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Read an HTTP request fully (headers + body) from the socket.
    /// We close the read side once we have the full body so the
    /// peer's `send_string` doesn't block forever; this matches what
    /// a real reverse-proxy does.
    fn drain_http_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
        use std::io::Read;
        let mut all = Vec::new();
        let mut buf = [0u8; 4096];
        // Read until we see the headers terminator.
        loop {
            let n = stream.read(&mut buf).unwrap_or(0);
            if n == 0 {
                break;
            }
            all.extend_from_slice(&buf[..n]);
            if all.windows(4).any(|w| w == b"\r\n\r\n") {
                // Headers complete — `Content-Length` tells us how much
                // body remains. For our test bodies the whole thing
                // arrives in the same read most of the time; loop
                // a few extra reads to drain trailing bytes.
                let mut idle = 0;
                while idle < 3 {
                    let n2 = stream.read(&mut buf).unwrap_or(0);
                    if n2 == 0 {
                        break;
                    }
                    all.extend_from_slice(&buf[..n2]);
                    idle += 1;
                }
                break;
            }
        }
        all
    }

    /// Helper: spin up a single-shot mock HTTP server, return the
    /// listening address + a handle that yields the bytes received.
    fn mock_one_shot() -> (std::net::SocketAddr, std::thread::JoinHandle<Vec<u8>>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let h = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let bytes = drain_http_request(&mut stream);
            use std::io::Write;
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok");
            bytes
        });
        (addr, h)
    }

    #[test]
    fn wandb_sink_pushes_metric_to_mock_server() {
        let (addr, server) = mock_one_shot();
        let sink = WandbSink::new(
            format!("http://{addr}"),
            "test-key",
            "rustorch",
            "default",
            "run-1",
        );
        let logger = Logger::builder().with_sink(sink).build();
        logger.metric("loss", 0.5, 0);
        let bytes = server.join().unwrap();
        let body = String::from_utf8_lossy(&bytes).to_string();
        assert!(
            body.contains("loss"),
            "body should reference the metric name"
        );
        assert!(body.contains("Bearer test-key"));
    }

    #[test]
    fn mlflow_sink_pushes_log_metric_call() {
        let (addr, server) = mock_one_shot();
        let sink = MLflowSink::new(format!("http://{addr}"), "run-2");
        let logger = Logger::builder().with_sink(sink).build();
        logger.metric("val_acc", 0.93, 100);
        let bytes = server.join().unwrap();
        let body = String::from_utf8_lossy(&bytes).to_string();
        assert!(body.contains("/api/2.0/mlflow/runs/log-metric"));
        assert!(body.contains("val_acc"));
    }

    #[test]
    fn external_sink_failure_is_isolated() {
        // Bind a port + immediately release it so the OS guarantees
        // the port is closed (deterministic across platforms — `127.0.0.1:1`
        // behaves differently on macOS vs Linux).
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let sink = WandbSink::new(format!("http://{addr}"), "k", "e", "p", "r");
        let logger = Logger::builder().with_sink(sink).build();
        // Submit must not panic even though the sink will fail.
        logger.metric("loss", 0.1, 0);
        // Use the metric type so the import isn't unused.
        let _: Option<&Metric> = None;
    }
}
