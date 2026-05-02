//! Autoscale controller. Inputs are the metrics the runtime
//! observes (RPS, p99 latency, queue depth, GPU util); outputs are
//! `ScaleDecision`s the orchestrator (Kubernetes HPA, Docker
//! Compose, plain process supervisor) acts on.
//!
//! The crate doesn't bind to k8s directly — instead we ship the
//! HPA YAML manifests as static files (`deploy/k8s/`) so the user
//! can `kubectl apply -f` them. The controller logic here is the
//! same shape Kubernetes uses internally; running it locally is
//! useful for Docker Compose setups + dry-run debugging.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Live observations from the serving runtime — what the
/// controller sees.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct LoadSnapshot {
    /// Current requests per second (rolling 30s average).
    pub rps: f64,
    /// 99th-percentile end-to-end latency in milliseconds.
    pub p99_latency_ms: f64,
    /// Current pending queue depth — the batcher's mpsc usage.
    pub queue_depth: usize,
    /// GPU utilisation 0–100, averaged across active workers.
    pub gpu_util: u32,
}

/// Operational target — what counts as "healthy". Defaults are
/// borrowed from doc v0.7.2 §"Throughput vs latency knobs".
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AutoscaleTarget {
    /// Maximum acceptable p99 latency. Above this we scale up.
    pub p99_target_ms: f64,
    /// Queue depth that triggers a scale-up regardless of latency.
    pub queue_threshold: usize,
    /// Idle window before a scale-down — protects against thrashing
    /// when traffic is bursty.
    pub idle_window: Duration,
    /// Minimum / maximum worker count.
    pub min_workers: u32,
    pub max_workers: u32,
    /// Step size for each scale event.
    pub step: u32,
}

impl Default for AutoscaleTarget {
    fn default() -> Self {
        Self {
            p99_target_ms: 100.0,
            queue_threshold: 64,
            idle_window: Duration::from_secs(300),
            min_workers: 1,
            max_workers: 16,
            step: 1,
        }
    }
}

/// What the controller asks the orchestrator to do.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScaleDecision {
    /// No change.
    Hold,
    /// Increase worker count by `delta`.
    ScaleUp { delta: u32 },
    /// Decrease worker count by `delta`.
    ScaleDown { delta: u32 },
}

/// Cooldown-aware autoscale controller. State is carried in
/// `current_workers` + the timestamp of the last scale event so the
/// caller can persist it across a restart if needed.
#[derive(Debug)]
pub struct AutoscaleController {
    pub target: AutoscaleTarget,
    pub current_workers: u32,
    last_scale: Option<Instant>,
    /// Wall-clock when the load was last considered "idle". `None`
    /// while we're still serving traffic.
    idle_since: Option<Instant>,
}

impl AutoscaleController {
    pub fn new(target: AutoscaleTarget, initial_workers: u32) -> Self {
        Self {
            target,
            current_workers: initial_workers.clamp(target.min_workers, target.max_workers),
            last_scale: None,
            idle_since: None,
        }
    }

    /// Feed a snapshot, get a decision back. The caller is
    /// expected to apply the decision (spawn / kill workers) and
    /// update `current_workers` accordingly via `apply_decision`.
    pub fn tick(&mut self, snap: LoadSnapshot) -> ScaleDecision {
        // Scale-up triggers (any one is sufficient).
        let latency_breach = snap.p99_latency_ms > self.target.p99_target_ms;
        let queue_breach = snap.queue_depth >= self.target.queue_threshold;
        if latency_breach || queue_breach {
            self.idle_since = None;
            return self.try_scale_up();
        }

        // Idle ≜ low GPU util AND empty queue. Sustained idle past
        // `idle_window` triggers a scale-down.
        let idle_now = snap.gpu_util < 5 && snap.queue_depth == 0;
        if idle_now {
            let started = self.idle_since.get_or_insert_with(Instant::now);
            if started.elapsed() >= self.target.idle_window {
                let dec = self.try_scale_down();
                if matches!(dec, ScaleDecision::ScaleDown { .. }) {
                    // Reset the idle timer so we don't scale down N
                    // times in a row.
                    self.idle_since = Some(Instant::now());
                }
                return dec;
            }
        } else {
            self.idle_since = None;
        }
        ScaleDecision::Hold
    }

    /// Apply a previously emitted decision to the internal counter.
    /// Real orchestrators may report back asynchronously; in tests
    /// the caller can call this immediately.
    pub fn apply_decision(&mut self, dec: ScaleDecision) {
        match dec {
            ScaleDecision::ScaleUp { delta } => {
                self.current_workers = (self.current_workers + delta).min(self.target.max_workers);
            },
            ScaleDecision::ScaleDown { delta } => {
                self.current_workers = self
                    .current_workers
                    .saturating_sub(delta)
                    .max(self.target.min_workers);
            },
            ScaleDecision::Hold => {},
        }
    }

    fn try_scale_up(&mut self) -> ScaleDecision {
        if self.current_workers >= self.target.max_workers {
            return ScaleDecision::Hold;
        }
        let delta = self
            .target
            .step
            .min(self.target.max_workers - self.current_workers);
        self.last_scale = Some(Instant::now());
        ScaleDecision::ScaleUp { delta }
    }

    fn try_scale_down(&mut self) -> ScaleDecision {
        if self.current_workers <= self.target.min_workers {
            return ScaleDecision::Hold;
        }
        let delta = self
            .target
            .step
            .min(self.current_workers - self.target.min_workers);
        self.last_scale = Some(Instant::now());
        ScaleDecision::ScaleDown { delta }
    }
}

/// Generate the Kubernetes HPA manifest for the given service. The
/// resulting YAML targets the runtime's `/metrics` endpoint via the
/// `prometheus-adapter` so HPA can read p99 latency.
pub fn render_k8s_hpa_manifest(name: &str, target: &AutoscaleTarget) -> String {
    format!(
        "apiVersion: autoscaling/v2\n\
         kind: HorizontalPodAutoscaler\n\
         metadata:\n  \
           name: {name}\n\
         spec:\n  \
           scaleTargetRef:\n    \
             apiVersion: apps/v1\n    \
             kind: Deployment\n    \
             name: {name}\n  \
           minReplicas: {min}\n  \
           maxReplicas: {max}\n  \
           metrics:\n  \
           - type: Pods\n    \
             pods:\n      \
               metric:\n        \
                 name: rustorch_serve_p99_latency_ms\n      \
               target:\n        \
                 type: AverageValue\n        \
                 averageValue: \"{p99}\"\n  \
           - type: Pods\n    \
             pods:\n      \
               metric:\n        \
                 name: rustorch_serve_queue_depth\n      \
               target:\n        \
                 type: AverageValue\n        \
                 averageValue: \"{q}\"\n",
        name = name,
        min = target.min_workers,
        max = target.max_workers,
        p99 = target.p99_target_ms as i64,
        q = target.queue_threshold,
    )
}

/// Static Docker Compose snippet — useful for dev environments that
/// want a 2-worker autoscale-style setup without a real cluster.
pub fn render_docker_compose() -> String {
    "version: '3.8'\n\
     services:\n  \
       rustorch-serve:\n    \
         image: rustorch/serve:latest\n    \
         ports:\n      \
           - \"8000:8000\"\n    \
         environment:\n      \
           - RUSTORCH_MAX_BATCH=32\n      \
           - RUSTORCH_MAX_WAIT_MS=20\n    \
         deploy:\n      \
           replicas: 2\n      \
           resources:\n        \
             limits:\n          \
               memory: 4G\n  \
       prometheus:\n    \
         image: prom/prometheus:latest\n    \
         ports:\n      \
           - \"9090:9090\"\n"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(rps: f64, p99: f64, queue: usize, util: u32) -> LoadSnapshot {
        LoadSnapshot {
            rps,
            p99_latency_ms: p99,
            queue_depth: queue,
            gpu_util: util,
        }
    }

    #[test]
    fn latency_breach_triggers_scale_up() {
        let mut c = AutoscaleController::new(AutoscaleTarget::default(), 1);
        let dec = c.tick(snap(50.0, 150.0, 4, 80));
        match dec {
            ScaleDecision::ScaleUp { delta } => assert_eq!(delta, 1),
            other => panic!("expected scale-up, got {other:?}"),
        }
    }

    #[test]
    fn queue_threshold_triggers_scale_up() {
        let mut c = AutoscaleController::new(AutoscaleTarget::default(), 1);
        let dec = c.tick(snap(10.0, 50.0, 100, 50));
        assert_eq!(dec, ScaleDecision::ScaleUp { delta: 1 });
    }

    #[test]
    fn no_breach_holds() {
        let mut c = AutoscaleController::new(AutoscaleTarget::default(), 4);
        let dec = c.tick(snap(50.0, 50.0, 8, 60));
        assert_eq!(dec, ScaleDecision::Hold);
    }

    #[test]
    fn max_workers_caps_scale_up() {
        let target = AutoscaleTarget {
            max_workers: 4,
            ..Default::default()
        };
        let mut c = AutoscaleController::new(target, 4);
        let dec = c.tick(snap(100.0, 200.0, 100, 90));
        assert_eq!(dec, ScaleDecision::Hold);
    }

    #[test]
    fn idle_window_must_elapse_before_scale_down() {
        let target = AutoscaleTarget {
            idle_window: Duration::from_millis(50),
            ..Default::default()
        };
        let mut c = AutoscaleController::new(target, 2);
        // First idle tick — start the timer, do not scale down yet.
        assert_eq!(c.tick(snap(0.0, 5.0, 0, 0)), ScaleDecision::Hold);
        std::thread::sleep(Duration::from_millis(60));
        // After the window, idle tick scales down.
        let dec = c.tick(snap(0.0, 5.0, 0, 0));
        assert_eq!(dec, ScaleDecision::ScaleDown { delta: 1 });
    }

    #[test]
    fn min_workers_floors_scale_down() {
        let target = AutoscaleTarget {
            min_workers: 1,
            idle_window: Duration::from_millis(0),
            ..Default::default()
        };
        let mut c = AutoscaleController::new(target, 1);
        std::thread::sleep(Duration::from_millis(1));
        let dec = c.tick(snap(0.0, 5.0, 0, 0));
        assert_eq!(dec, ScaleDecision::Hold);
    }

    #[test]
    fn apply_decision_updates_counter() {
        let mut c = AutoscaleController::new(AutoscaleTarget::default(), 1);
        c.apply_decision(ScaleDecision::ScaleUp { delta: 2 });
        assert_eq!(c.current_workers, 3);
        c.apply_decision(ScaleDecision::ScaleDown { delta: 1 });
        assert_eq!(c.current_workers, 2);
    }

    #[test]
    fn renders_valid_hpa_yaml_with_targets() {
        let m = render_k8s_hpa_manifest(
            "rustorch-serve",
            &AutoscaleTarget {
                p99_target_ms: 100.0,
                queue_threshold: 64,
                min_workers: 1,
                max_workers: 8,
                ..Default::default()
            },
        );
        assert!(m.contains("kind: HorizontalPodAutoscaler"));
        assert!(m.contains("minReplicas: 1"));
        assert!(m.contains("maxReplicas: 8"));
        assert!(m.contains("rustorch_serve_p99_latency_ms"));
        assert!(m.contains("rustorch_serve_queue_depth"));
    }

    #[test]
    fn docker_compose_template_lists_services() {
        let s = render_docker_compose();
        assert!(s.contains("rustorch-serve"));
        assert!(s.contains("prometheus"));
        assert!(s.contains("RUSTORCH_MAX_BATCH"));
    }
}
