//! Training-loop checkpoint helper (P1.8).
//!
//! Builder-style: configure cadence, retention, and "on-improve"
//! semantics, then call `step(epoch, step, metric, sd)` from the
//! training loop. The helper writes safetensors files to the configured
//! directory and rotates old checkpoints.
//!
//! Example:
//! ```ignore
//! let mut ckpt = Checkpoint::new("runs/exp1")
//!     .every(Cadence::Epochs(1))
//!     .keep(5)
//!     .on_improve("val_acc");
//!
//! for epoch in 0..n_epochs {
//!     // ... train / eval ...
//!     let mut metrics = HashMap::new();
//!     metrics.insert("val_acc".into(), val_acc);
//!     metrics.insert("val_loss".into(), val_loss);
//!     let sd = state_dict(&model);
//!     ckpt.step(epoch, step, &metrics, &sd)?;
//! }
//! ```

use crate::SafetensorsError;
use rustorch_core::tensor::tensor_impl::Tensor;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// When to fire a save.
#[derive(Debug, Clone, Copy)]
pub enum Cadence {
    /// Save every N epochs (default 1).
    Epochs(usize),
    /// Save every N optimisation steps.
    Steps(usize),
    /// Save every wall-clock interval.
    WallClock(Duration),
}

/// Whether higher or lower is better for the on-improve metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Save only when the metric increases (e.g. accuracy).
    Maximize,
    /// Save only when the metric decreases (e.g. loss).
    Minimize,
}

/// Training-loop checkpoint builder.
pub struct Checkpoint {
    save_dir: PathBuf,
    cadence: Cadence,
    keep: usize,
    on_improve: Option<(String, Direction)>,

    last_saved_epoch: Option<usize>,
    last_saved_step: Option<usize>,
    last_saved_at: Option<Instant>,
    best_metric: Option<f32>,
    history: VecDeque<PathBuf>,
}

impl Checkpoint {
    /// New checkpoint helper that writes to `save_dir`. Defaults:
    /// `Cadence::Epochs(1)`, `keep(0)` (unlimited), no on_improve.
    pub fn new<P: Into<PathBuf>>(save_dir: P) -> Self {
        Checkpoint {
            save_dir: save_dir.into(),
            cadence: Cadence::Epochs(1),
            keep: 0,
            on_improve: None,
            last_saved_epoch: None,
            last_saved_step: None,
            last_saved_at: None,
            best_metric: None,
            history: VecDeque::new(),
        }
    }

    /// Set the firing cadence.
    #[must_use]
    pub fn every(mut self, cadence: Cadence) -> Self {
        self.cadence = cadence;
        self
    }

    /// Keep at most `n` rotated checkpoints (older ones get unlinked).
    /// `0` means keep all.
    #[must_use]
    pub fn keep(mut self, n: usize) -> Self {
        self.keep = n;
        self
    }

    /// Only save when `metric` reaches a new best (Maximize default).
    #[must_use]
    pub fn on_improve<S: Into<String>>(mut self, metric: S) -> Self {
        self.on_improve = Some((metric.into(), Direction::Maximize));
        self
    }

    /// Override on_improve direction.
    #[must_use]
    pub fn on_improve_direction(mut self, direction: Direction) -> Self {
        if let Some((m, _)) = self.on_improve.take() {
            self.on_improve = Some((m, direction));
        }
        self
    }

    /// Should we save right now? (Pure check; doesn't write.)
    pub fn should_save(&self, epoch: usize, step: usize, metrics: &HashMap<String, f32>) -> bool {
        // Cadence first.
        let cadence_ok = match self.cadence {
            Cadence::Epochs(n) => match self.last_saved_epoch {
                Some(prev) => epoch >= prev + n,
                None => true,
            },
            Cadence::Steps(n) => match self.last_saved_step {
                Some(prev) => step >= prev + n,
                None => true,
            },
            Cadence::WallClock(d) => match self.last_saved_at {
                Some(prev) => prev.elapsed() >= d,
                None => true,
            },
        };
        if !cadence_ok {
            return false;
        }
        // on-improve gate.
        if let Some((metric, dir)) = &self.on_improve {
            let v = match metrics.get(metric) {
                Some(&v) => v,
                None => {
                    // Missing metric → still save (with warning in logs);
                    // matches PyTorch's tolerance.
                    return true;
                },
            };
            if let Some(best) = self.best_metric {
                let is_better = match dir {
                    Direction::Maximize => v > best,
                    Direction::Minimize => v < best,
                };
                return is_better;
            }
            return true; // first time, always save.
        }
        true
    }

    /// One step of the training loop. Decides if a save is due, then
    /// writes the safetensors file and rotates history.
    pub fn step(
        &mut self,
        epoch: usize,
        step: usize,
        metrics: &HashMap<String, f32>,
        sd: &BTreeMap<String, Tensor>,
    ) -> Result<Option<PathBuf>, SafetensorsError> {
        if !self.should_save(epoch, step, metrics) {
            return Ok(None);
        }
        std::fs::create_dir_all(&self.save_dir).map_err(SafetensorsError::Io)?;
        let path = self
            .save_dir
            .join(format!("ckpt_epoch{epoch}_step{step}.safetensors"));
        crate::safetensors::write_path(&path, sd)?;

        // Update tracking state.
        self.last_saved_epoch = Some(epoch);
        self.last_saved_step = Some(step);
        self.last_saved_at = Some(Instant::now());
        if let Some((metric, dir)) = &self.on_improve {
            if let Some(&v) = metrics.get(metric) {
                let new_best = match (self.best_metric, dir) {
                    (None, _) => true,
                    (Some(b), Direction::Maximize) => v > b,
                    (Some(b), Direction::Minimize) => v < b,
                };
                if new_best {
                    self.best_metric = Some(v);
                }
            }
        }

        // Rotation: keep last `self.keep` (0 = unlimited).
        self.history.push_back(path.clone());
        if self.keep > 0 {
            while self.history.len() > self.keep {
                if let Some(old) = self.history.pop_front() {
                    let _ = std::fs::remove_file(&old);
                }
            }
        }
        Ok(Some(path))
    }

    /// Number of files currently in the rotation history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_core::tensor::tensor_impl::Tensor;
    use std::collections::HashMap;

    fn dummy_sd() -> BTreeMap<String, Tensor> {
        let mut m = BTreeMap::new();
        m.insert(
            "weight".to_string(),
            Tensor::from_vec([2usize], vec![1.0_f32, 2.0]).unwrap(),
        );
        m
    }

    #[test]
    fn epochs_cadence_fires_each_n_epochs() {
        let dir = tempdir();
        let mut ckpt = Checkpoint::new(&dir).every(Cadence::Epochs(2));
        let metrics = HashMap::new();
        let sd = dummy_sd();
        // Epoch 0 → fires (first time).
        assert!(ckpt.step(0, 0, &metrics, &sd).unwrap().is_some());
        // Epoch 1 → no.
        assert!(ckpt.step(1, 0, &metrics, &sd).unwrap().is_none());
        // Epoch 2 → fires.
        assert!(ckpt.step(2, 0, &metrics, &sd).unwrap().is_some());
    }

    #[test]
    fn steps_cadence_fires_each_n_steps() {
        let dir = tempdir();
        let mut ckpt = Checkpoint::new(&dir).every(Cadence::Steps(5));
        let metrics = HashMap::new();
        let sd = dummy_sd();
        assert!(ckpt.step(0, 0, &metrics, &sd).unwrap().is_some());
        assert!(ckpt.step(0, 4, &metrics, &sd).unwrap().is_none());
        assert!(ckpt.step(0, 5, &metrics, &sd).unwrap().is_some());
    }

    #[test]
    fn keep_n_rotates_old_files() {
        let dir = tempdir();
        let mut ckpt = Checkpoint::new(&dir).every(Cadence::Epochs(1)).keep(3);
        let metrics = HashMap::new();
        let sd = dummy_sd();
        for ep in 0..6 {
            ckpt.step(ep, 0, &metrics, &sd).unwrap();
        }
        // Should keep the last 3 only.
        assert_eq!(ckpt.history_len(), 3);
    }

    #[test]
    fn on_improve_only_saves_when_metric_better() {
        let dir = tempdir();
        let mut ckpt = Checkpoint::new(&dir)
            .every(Cadence::Epochs(1))
            .on_improve("val_acc");
        let mut metrics = HashMap::new();
        let sd = dummy_sd();
        metrics.insert("val_acc".to_string(), 0.5);
        assert!(ckpt.step(0, 0, &metrics, &sd).unwrap().is_some());
        // No improvement → no save.
        metrics.insert("val_acc".to_string(), 0.4);
        assert!(ckpt.step(1, 0, &metrics, &sd).unwrap().is_none());
        // Improvement → save.
        metrics.insert("val_acc".to_string(), 0.6);
        assert!(ckpt.step(2, 0, &metrics, &sd).unwrap().is_some());
    }

    #[test]
    fn on_improve_minimize_for_loss() {
        let dir = tempdir();
        let mut ckpt = Checkpoint::new(&dir)
            .every(Cadence::Epochs(1))
            .on_improve("val_loss")
            .on_improve_direction(Direction::Minimize);
        let mut metrics = HashMap::new();
        let sd = dummy_sd();
        metrics.insert("val_loss".to_string(), 1.0);
        assert!(ckpt.step(0, 0, &metrics, &sd).unwrap().is_some());
        metrics.insert("val_loss".to_string(), 1.5);
        assert!(ckpt.step(1, 0, &metrics, &sd).unwrap().is_none());
        metrics.insert("val_loss".to_string(), 0.5);
        assert!(ckpt.step(2, 0, &metrics, &sd).unwrap().is_some());
    }

    #[test]
    fn missing_metric_still_saves() {
        let dir = tempdir();
        let mut ckpt = Checkpoint::new(&dir)
            .every(Cadence::Epochs(1))
            .on_improve("never_logged");
        let metrics = HashMap::new();
        let sd = dummy_sd();
        // Even with on_improve, missing metric → save (warning behavior).
        assert!(ckpt.step(0, 0, &metrics, &sd).unwrap().is_some());
    }

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rustorch_ckpt_test_{}", uniq_id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn uniq_id() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }
}
