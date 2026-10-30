// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-training.md, section "config.rs"
// ============================================================================
//! The settings of a training run.

use midman_foundation::{Error, Result};
use midman_optim::{AdamW, WarmupCosine};
use std::collections::BTreeMap;

/// Everything about a run that is not the model or the data.
///
/// These settings are saved with the state of a run and checked when it is resumed: a run
/// that continues with other settings is not the same run, and is refused rather than
/// continued silently.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    /// Sequences in a batch.
    pub batch_size: usize,
    /// Tokens in a sequence the model sees (a window holds one more, the last target).
    pub seq_len: usize,
    /// Seed of the order in which the data is visited.
    pub seed: u64,
    /// The learning rate at the end of warmup.
    pub max_lr: f32,
    /// The learning rate the cosine decay ends at, and stays at.
    pub min_lr: f32,
    /// Steps of linear warmup.
    pub warmup_steps: u64,
    /// The step at which the schedule reaches `min_lr`.
    pub total_steps: u64,
    /// AdamW's first-moment decay.
    pub beta1: f32,
    /// AdamW's second-moment decay.
    pub beta2: f32,
    /// AdamW's epsilon.
    pub eps: f32,
    /// AdamW's weight decay (applied to matrices and embeddings only).
    pub weight_decay: f32,
    /// The gradient is rescaled so that its global norm is at most this.
    pub clip_norm: f32,
}

impl TrainConfig {
    /// Settings for `total_steps` steps of batches of `batch_size` sequences of `seq_len`
    /// tokens, with defaults for the rest: `max_lr` 1e-3, `min_lr` 1e-4, warmup over a
    /// twentieth of the steps, betas 0.9 and 0.95, epsilon 1e-8, weight decay 0.1 and
    /// clipping at 1.0. None of these has been tuned.
    pub fn new(batch_size: usize, seq_len: usize, total_steps: u64) -> TrainConfig {
        TrainConfig {
            batch_size,
            seq_len,
            seed: 0,
            max_lr: 1e-3,
            min_lr: 1e-4,
            warmup_steps: (total_steps / 20).min(total_steps.saturating_sub(1)),
            total_steps,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            clip_norm: 1.0,
        }
    }

    /// Checks the settings.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch size, the sequence length or the number of steps is
    /// zero, if the clipping norm is not positive and finite, or for any reason that
    /// [`WarmupCosine::new`] or [`AdamW::new`] refuse.
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 || self.seq_len == 0 || self.total_steps == 0 {
            return Err(Error::invalid(format!(
                "training: batch_size ({}), seq_len ({}) and total_steps ({}) must all be above zero",
                self.batch_size, self.seq_len, self.total_steps
            )));
        }
        if !(self.clip_norm.is_finite() && self.clip_norm > 0.0) {
            return Err(Error::invalid(format!(
                "training: clip_norm must be positive and finite, got {}",
                self.clip_norm
            )));
        }
        self.schedule()?;
        self.optimizer()?;
        Ok(())
    }

    /// The learning-rate schedule of these settings.
    ///
    /// # Errors
    ///
    /// Returns an error if [`WarmupCosine::new`] refuses them.
    pub fn schedule(&self) -> Result<WarmupCosine> {
        WarmupCosine::new(self.max_lr, self.min_lr, self.warmup_steps, self.total_steps)
    }

    /// A new optimizer with these settings and no state.
    ///
    /// # Errors
    ///
    /// Returns an error if [`AdamW::new`] refuses them.
    pub fn optimizer(&self) -> Result<AdamW> {
        AdamW::new(self.beta1, self.beta2, self.eps, self.weight_decay)
    }

    /// The settings as `(key, text)` pairs. A float is written the shortest way that reads
    /// back as the same number.
    fn entries(&self) -> [(&'static str, String); 12] {
        [
            ("train.batch_size", self.batch_size.to_string()),
            ("train.seq_len", self.seq_len.to_string()),
            ("train.seed", self.seed.to_string()),
            ("train.max_lr", self.max_lr.to_string()),
            ("train.min_lr", self.min_lr.to_string()),
            ("train.warmup_steps", self.warmup_steps.to_string()),
            ("train.total_steps", self.total_steps.to_string()),
            ("train.beta1", self.beta1.to_string()),
            ("train.beta2", self.beta2.to_string()),
            ("train.eps", self.eps.to_string()),
            ("train.weight_decay", self.weight_decay.to_string()),
            ("train.clip_norm", self.clip_norm.to_string()),
        ]
    }

    /// The settings as file metadata.
    pub fn to_metadata(&self) -> BTreeMap<String, String> {
        self.entries().into_iter().map(|(key, text)| (key.to_string(), text)).collect()
    }

    /// Checks that `stored` holds exactly these settings.
    ///
    /// # Errors
    ///
    /// Returns an error naming every setting that is missing from `stored` or differs.
    pub fn check_matches(&self, stored: &BTreeMap<String, String>) -> Result<()> {
        let mut problems = Vec::new();
        for (key, now) in self.entries() {
            match stored.get(key) {
                None => problems.push(format!("`{key}` is missing from the saved run")),
                Some(then) if *then != now => {
                    problems.push(format!("`{key}` was {then} and is now {now}"));
                }
                Some(_) => {}
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(Error::invalid(format!(
                "the saved run was made with other settings: {}",
                problems.join("; ")
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_valid_and_follow_the_step_count() {
        for steps in [1, 2, 19, 20, 21, 1000] {
            let config = TrainConfig::new(4, 8, steps);
            config.validate().unwrap();
            assert_eq!(config.total_steps, steps);
            assert!(config.warmup_steps < steps);
        }
        assert_eq!(TrainConfig::new(4, 8, 1000).warmup_steps, 50);
        assert_eq!(TrainConfig::new(4, 8, 1).warmup_steps, 0);
    }

    #[test]
    fn each_bad_setting_is_refused() {
        let good = TrainConfig::new(4, 8, 100);
        type Change = Box<dyn Fn(&mut TrainConfig)>;
        let cases: Vec<(&str, Change)> = vec![
            ("batch", Box::new(|c| c.batch_size = 0)),
            ("seq", Box::new(|c| c.seq_len = 0)),
            ("steps", Box::new(|c| c.total_steps = 0)),
            ("clip zero", Box::new(|c| c.clip_norm = 0.0)),
            ("clip negative", Box::new(|c| c.clip_norm = -1.0)),
            ("clip nan", Box::new(|c| c.clip_norm = f32::NAN)),
            ("clip infinite", Box::new(|c| c.clip_norm = f32::INFINITY)),
            ("max_lr", Box::new(|c| c.max_lr = 0.0)),
            ("min above max", Box::new(|c| c.min_lr = c.max_lr * 2.0)),
            ("warmup", Box::new(|c| c.warmup_steps = c.total_steps)),
            ("beta1", Box::new(|c| c.beta1 = 1.0)),
            ("beta2", Box::new(|c| c.beta2 = -0.1)),
            ("eps", Box::new(|c| c.eps = 0.0)),
            ("decay", Box::new(|c| c.weight_decay = -1.0)),
        ];
        for (name, change) in cases {
            let mut config = good.clone();
            change(&mut config);
            assert!(config.validate().is_err(), "{name} was accepted");
        }
        good.validate().unwrap();
    }

    #[test]
    fn the_metadata_reads_back_as_the_same_settings() {
        let mut config = TrainConfig::new(3, 5, 77);
        config.seed = u64::MAX;
        config.max_lr = 0.003_141_592_7;
        config.eps = 1e-7;
        config.weight_decay = 0.0;
        let metadata = config.to_metadata();
        assert_eq!(metadata.len(), 12);
        assert_eq!(metadata["train.seed"], u64::MAX.to_string());
        config.check_matches(&metadata).unwrap();
        // A float comes back as the same number, not a near one.
        let back: f32 = metadata["train.max_lr"].parse().unwrap();
        assert_eq!(back.to_bits(), config.max_lr.to_bits());
    }

    #[test]
    fn every_difference_is_named() {
        let config = TrainConfig::new(4, 8, 100);
        let mut stored = config.to_metadata();
        config.check_matches(&stored).unwrap();

        let mut other = config.clone();
        other.max_lr = 2e-3;
        other.seed = 9;
        let err = other.check_matches(&stored).unwrap_err().to_string();
        assert!(err.contains("`train.max_lr` was 0.001 and is now 0.002"), "{err}");
        assert!(err.contains("`train.seed` was 0 and is now 9"), "{err}");
        assert!(!err.contains("batch_size"), "{err}");

        stored.remove("train.clip_norm");
        let err = config.check_matches(&stored).unwrap_err().to_string();
        assert!(err.contains("`train.clip_norm` is missing"), "{err}");

        // Settings that are only in the file and not in the run are not a difference.
        let mut extra = config.to_metadata();
        extra.insert("anything.else".to_string(), "x".to_string());
        config.check_matches(&extra).unwrap();
    }

    #[test]
    fn the_schedule_and_the_optimizer_follow_the_settings() {
        let mut config = TrainConfig::new(4, 8, 100);
        config.max_lr = 0.5;
        config.min_lr = 0.05;
        config.warmup_steps = 10;
        let schedule = config.schedule().unwrap();
        assert_eq!(schedule.total_steps(), 100);
        assert_eq!(schedule.lr(9), 0.5);
        assert_eq!(schedule.lr(100), 0.05);
        config.weight_decay = 0.3;
        assert_eq!(config.optimizer().unwrap().hyperparameters(), (0.9, 0.95, 1e-8, 0.3));
    }
}
