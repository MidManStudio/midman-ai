// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-optim.md, section "schedule.rs"
// ============================================================================
//! Learning-rate schedule: linear warmup, then cosine decay.

use midman_foundation::{Error, Result};

/// Linear warmup to `max_lr`, then cosine decay to `min_lr`, then constant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarmupCosine {
    max_lr: f32,
    min_lr: f32,
    warmup_steps: u64,
    total_steps: u64,
}

impl WarmupCosine {
    /// Creates a schedule. `warmup_steps` may be 0.
    ///
    /// # Errors
    ///
    /// Returns an error unless `0 <= min_lr <= max_lr`, `max_lr` is positive and
    /// finite, and `warmup_steps < total_steps`.
    pub fn new(
        max_lr: f32,
        min_lr: f32,
        warmup_steps: u64,
        total_steps: u64,
    ) -> Result<WarmupCosine> {
        if !(max_lr.is_finite() && max_lr > 0.0) {
            return Err(Error::invalid(format!(
                "schedule: max_lr must be positive and finite, got {max_lr}"
            )));
        }
        if !(min_lr.is_finite() && min_lr >= 0.0 && min_lr <= max_lr) {
            return Err(Error::invalid(format!(
                "schedule: min_lr must be in [0, max_lr = {max_lr}], got {min_lr}"
            )));
        }
        if warmup_steps >= total_steps {
            return Err(Error::invalid(format!(
                "schedule: warmup_steps {warmup_steps} must be below total_steps {total_steps}"
            )));
        }
        Ok(WarmupCosine { max_lr, min_lr, warmup_steps, total_steps })
    }

    /// The learning rate for the optimizer step with zero-based index `step`.
    ///
    /// During warmup it is `max_lr * (step + 1) / warmup_steps`, so the first
    /// step already moves. It equals `max_lr` at step `warmup_steps` and decays
    /// along half a cosine to `min_lr` at step `total_steps`, then stays there.
    pub fn lr(&self, step: u64) -> f32 {
        let (max_lr, min_lr) = (f64::from(self.max_lr), f64::from(self.min_lr));
        if step < self.warmup_steps {
            return (max_lr * (step + 1) as f64 / self.warmup_steps as f64) as f32;
        }
        if step >= self.total_steps {
            return self.min_lr;
        }
        let progress =
            (step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;
        (min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())) as f32
    }

    /// The step at which the schedule reaches `min_lr` and stays.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule() -> WarmupCosine {
        WarmupCosine::new(1e-3, 1e-4, 4, 14).unwrap()
    }

    #[test]
    fn values_match_an_independent_calculation() {
        // Computed separately in Python (float64).
        let s = schedule();
        let expected = [
            (0, 2.5e-4),
            (1, 5.0e-4),
            (3, 1.0e-3),
            (4, 1.0e-3),
            (5, 9.779_75e-4),
            (9, 5.5e-4),
            (13, 1.220_25e-4),
            (14, 1.0e-4),
            (100, 1.0e-4),
        ];
        for (step, want) in expected {
            let got = s.lr(step);
            assert!((got - want).abs() < 1e-8, "step {step}: {got} vs {want}");
        }
    }

    #[test]
    fn warmup_rises_and_decay_never_rises() {
        let s = schedule();
        for step in 0..3 {
            assert!(s.lr(step) < s.lr(step + 1));
        }
        for step in 4..20 {
            assert!(s.lr(step + 1) <= s.lr(step) + 1e-12, "step {step}");
        }
    }

    #[test]
    fn no_warmup_starts_at_the_peak() {
        let s = WarmupCosine::new(0.5, 0.0, 0, 10).unwrap();
        assert_eq!(s.lr(0), 0.5);
        assert_eq!(s.lr(10), 0.0);
        assert_eq!(s.total_steps(), 10);
    }

    #[test]
    fn bad_arguments_are_errors() {
        assert!(WarmupCosine::new(0.0, 0.0, 0, 10).is_err());
        assert!(WarmupCosine::new(f32::NAN, 0.0, 0, 10).is_err());
        assert!(WarmupCosine::new(1.0, -0.1, 0, 10).is_err());
        assert!(WarmupCosine::new(1.0, 2.0, 0, 10).is_err());
        assert!(WarmupCosine::new(1.0, 0.0, 10, 10).is_err());
        assert!(WarmupCosine::new(1.0, 0.0, 11, 10).is_err());
        assert!(WarmupCosine::new(1.0, 0.0, 0, 0).is_err());
    }
}
