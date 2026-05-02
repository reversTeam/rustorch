//! Cluster telemetry source + 1Hz publisher task.
//!
//! Two providers ship today:
//!   * `MockClusterProvider` — always available, deterministic
//!     samples that drift slightly per tick so the dashboard sees
//!     motion. Used by every CI run + macOS dev box.
//!   * `NvmlClusterProvider` — Linux/Windows + the `nvml` Cargo
//!     feature. Wraps `nvml-wrapper` to surface real per-GPU util,
//!     vram, temp, power.
//!
//! `spawn_tick_task` runs the publisher loop. It holds an `Arc<dyn
//! ClusterProvider>` and a `Hub` handle, ticks every `interval`
//! and publishes the snapshot on the `cluster.tick` topic.
//!
//! The producer is intentionally separate from the route handler in
//! `handlers/cluster.rs` — the same `MockClusterProvider` is reused
//! by both `GET /cluster/gpus` (sync sample) and the SSE tick task
//! (1Hz fan-out) so they stay in lock-step.

use crate::sse::Hub;
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;

/// Topic emitted by `spawn_tick_task`. Frontend subscribes via
/// `GET /sse/cluster.tick`.
pub const TICK_TOPIC: &str = "cluster.tick";

/// Wire shape of a single GPU sample. Same struct as
/// `handlers/cluster::GpuSample` — re-exported here so the provider
/// trait can stay independent of the HTTP layer.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct GpuSample {
    pub id: u32,
    /// Compute utilization 0-100.
    pub util: u32,
    /// VRAM used in MiB.
    pub vram_used_mb: u32,
    /// VRAM total in MiB.
    pub vram_total_mb: u32,
    /// Edge temperature in °C.
    pub temp_c: u32,
    /// Power draw in watts.
    pub power_w: u32,
    /// Run IDs currently bound to this GPU (filled in by handlers
    /// from the run registry; provider returns an empty vec).
    pub runs: Vec<String>,
}

/// What every cluster provider must answer. Synchronous on purpose —
/// the providers do their I/O behind a mutex, the tick task wraps
/// the call in `tokio::task::spawn_blocking` if necessary.
pub trait ClusterProvider: Send + Sync {
    fn samples(&self) -> Vec<GpuSample>;
}

// ---- mock provider --------------------------------------------------

/// Deterministic provider that drifts each sample by a small step
/// per tick so the dashboard sparkline has signal even on a CI box.
pub struct MockClusterProvider {
    tick: std::sync::atomic::AtomicU32,
}

impl Default for MockClusterProvider {
    fn default() -> Self {
        Self {
            tick: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

impl MockClusterProvider {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ClusterProvider for MockClusterProvider {
    fn samples(&self) -> Vec<GpuSample> {
        // Roll a counter forward; util & power oscillate, vram and
        // temp drift slowly. Deterministic so tests stay stable.
        let t = self.tick.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        (0..2)
            .map(|i| {
                let phase = (t + i * 13) % 60;
                let util = ((phase as f32 / 60.0) * 100.0).round() as u32;
                GpuSample {
                    id: i,
                    util,
                    vram_used_mb: 256 + (t % 1024),
                    vram_total_mb: 24_564,
                    temp_c: 38 + (t % 25),
                    power_w: 60 + util,
                    runs: vec![],
                }
            })
            .collect()
    }
}

// ---- nvml provider --------------------------------------------------

#[cfg(feature = "nvml")]
pub mod nvml_impl {
    //! Real telemetry via libnvidia-ml. Loaded once at construction
    //! time; subsequent `samples()` calls go through cheap NVML
    //! getters. Errors are logged but never panic — a transient NVML
    //! glitch falls back to an empty sample list rather than killing
    //! the producer task.

    use super::{ClusterProvider, GpuSample};
    use nvml_wrapper::Nvml;
    use std::sync::Mutex;

    pub struct NvmlClusterProvider {
        nvml: Mutex<Nvml>,
    }

    impl NvmlClusterProvider {
        /// Initialise NVML. Returns `None` if the driver / library
        /// is missing (typical on a laptop without NVIDIA hardware
        /// or in a container that didn't get the device passthrough).
        pub fn try_new() -> Option<Self> {
            match Nvml::init() {
                Ok(nvml) => Some(Self {
                    nvml: Mutex::new(nvml),
                }),
                Err(e) => {
                    tracing::warn!(error = %e, "NVML init failed; falling back to mock");
                    None
                },
            }
        }
    }

    impl ClusterProvider for NvmlClusterProvider {
        fn samples(&self) -> Vec<GpuSample> {
            let nvml = self.nvml.lock().unwrap();
            let count = nvml.device_count().unwrap_or(0);
            (0..count)
                .filter_map(|i| {
                    let dev = nvml.device_by_index(i).ok()?;
                    let util = dev.utilization_rates().ok()?;
                    let mem = dev.memory_info().ok()?;
                    let temp = dev
                        .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                        .ok()
                        .unwrap_or(0);
                    let power = dev.power_usage().ok().unwrap_or(0) / 1000;
                    Some(GpuSample {
                        id: i,
                        util: util.gpu,
                        vram_used_mb: (mem.used / 1_048_576) as u32,
                        vram_total_mb: (mem.total / 1_048_576) as u32,
                        temp_c: temp,
                        power_w: power,
                        runs: vec![],
                    })
                })
                .collect()
        }
    }
}

/// Pick the best provider available on the current platform. NVML
/// when the feature is on AND the driver is reachable, otherwise the
/// deterministic mock.
pub fn default_provider() -> Arc<dyn ClusterProvider> {
    #[cfg(feature = "nvml")]
    {
        if let Some(p) = nvml_impl::NvmlClusterProvider::try_new() {
            tracing::info!("cluster: using NVML provider");
            return Arc::new(p);
        }
    }
    tracing::info!("cluster: using mock provider");
    Arc::new(MockClusterProvider::new())
}

// ---- 1Hz publisher --------------------------------------------------

/// Spawn a background task that polls `provider` every `interval`
/// and publishes the snapshot on the `cluster.tick` topic. Returns
/// the join handle so callers can shut it down explicitly (mostly
/// for tests; production lets it run for the process lifetime).
pub fn spawn_tick_task(
    hub: Hub,
    provider: Arc<dyn ClusterProvider>,
    interval: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        // Skip the first immediate tick; first publish lands one
        // `interval` after spawn so subscribers have time to attach.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker.tick().await; // discard the first instant tick
        loop {
            ticker.tick().await;
            let samples = provider.samples();
            hub.publish(
                TICK_TOPIC,
                "cluster.tick",
                &serde_json::json!({"gpus": samples}),
            );
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_provider_returns_two_samples_with_expected_shape() {
        let p = MockClusterProvider::new();
        let s = p.samples();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].id, 0);
        assert_eq!(s[1].id, 1);
        for g in &s {
            assert!(g.util <= 100);
            assert!(g.vram_total_mb > g.vram_used_mb);
        }
    }

    #[test]
    fn mock_provider_drifts_per_tick() {
        let p = MockClusterProvider::new();
        let a = p.samples();
        let b = p.samples();
        let c = p.samples();
        // The util cycle is 60 ticks long, so any 3 consecutive
        // ticks will produce different util values for at least one
        // GPU.
        assert!(
            a[0].util != b[0].util || b[0].util != c[0].util,
            "expected drift across ticks"
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn tick_task_publishes_at_interval() {
        let hub = Hub::new();
        let provider = Arc::new(MockClusterProvider::new());
        let handle = spawn_tick_task(hub.clone(), provider, Duration::from_millis(50));

        // Advance the paused clock by 200ms → 4 ticks expected, of
        // which the first is discarded inside the task.
        tokio::time::advance(Duration::from_millis(200)).await;
        // Yield so the spawned task can flush its publishes.
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_millis(50)).await;
        tokio::task::yield_now().await;

        // The hub ring should now hold at least one cluster.tick event.
        assert!(
            hub.ring_len(TICK_TOPIC) >= 1,
            "expected at least one tick to be buffered"
        );

        handle.abort();
    }
}
