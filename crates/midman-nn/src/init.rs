// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-nn.md, section "init.rs"
// ============================================================================
//! Standard deviations for weight initialization.

use midman_foundation::{Error, Result};

/// Standard deviations used to draw initial weights from a normal distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Init {
    /// Standard deviation for ordinary weights (embeddings, attention inputs,
    /// the MLP gate and up projections, the output head).
    pub std: f32,
    /// Standard deviation for the projections that write into the residual
    /// stream (the attention output and the MLP down projection).
    pub out_std: f32,
}

impl Init {
    /// The same standard deviation everywhere.
    pub fn uniform(std: f32) -> Init {
        Init { std, out_std: std }
    }

    /// `std` for ordinary weights, and `std / sqrt(2 * n_layers)` for the
    /// residual-stream projections, so the variance of the residual stream
    /// does not grow with depth.
    ///
    /// # Errors
    ///
    /// Returns an error if `std` is not positive and finite, or `n_layers` is 0.
    pub fn scaled(std: f32, n_layers: usize) -> Result<Init> {
        check_std("init", std)?;
        if n_layers == 0 {
            return Err(Error::invalid("init: n_layers must be at least 1"));
        }
        let out_std = std / (2.0 * n_layers as f32).sqrt();
        Ok(Init { std, out_std })
    }
}

/// Rejects a standard deviation that is not positive and finite.
pub(crate) fn check_std(what: &str, std: f32) -> Result<()> {
    if std.is_finite() && std > 0.0 {
        Ok(())
    } else {
        Err(Error::invalid(format!("{what}: init std must be positive and finite, got {std}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_shrinks_the_residual_projections_with_depth() {
        let one = Init::scaled(0.02, 1).unwrap();
        let eight = Init::scaled(0.02, 8).unwrap();
        assert_eq!(one.std, 0.02);
        assert!((one.out_std - 0.02 / 2f32.sqrt()).abs() < 1e-9);
        assert!((eight.out_std - 0.02 / 4.0).abs() < 1e-9);
    }

    #[test]
    fn uniform_uses_one_value() {
        let init = Init::uniform(0.5);
        assert_eq!((init.std, init.out_std), (0.5, 0.5));
    }

    #[test]
    fn bad_values_are_errors() {
        assert!(Init::scaled(0.0, 2).is_err());
        assert!(Init::scaled(-1.0, 2).is_err());
        assert!(Init::scaled(f32::NAN, 2).is_err());
        assert!(Init::scaled(0.02, 0).is_err());
    }
}
