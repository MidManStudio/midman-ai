// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-api.md, section "report.rs"
// ============================================================================
//! Answers that need no model: what a configuration costs, and the rotary tables.

use midman_foundation::Result;
use midman_model_config::{ModelConfig, ParamBreakdown};
use midman_nn::RotaryTables;

/// What a model configuration works out to.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigReport {
    /// The size of one attention head.
    pub head_dim: usize,
    /// The width of the key and value projections.
    pub kv_dim: usize,
    /// Parameters by category.
    pub params: ParamBreakdown,
}

/// Validates `config` and counts its parameters, without building anything.
///
/// # Errors
///
/// Returns the validation error if the configuration is invalid.
pub fn config_report(config: &ModelConfig) -> Result<ConfigReport> {
    config.validate()?;
    Ok(ConfigReport {
        head_dim: config.head_dim(),
        kv_dim: config.kv_dim(),
        params: config.param_breakdown()?,
    })
}

/// The rotary rotation tables as `(positions, half, cos, sin)`; `cos` and `sin`
/// are `positions * half` values each, one row per position.
///
/// # Errors
///
/// Returns an error if `head_dim` is 0 or odd, `max_seq_len` is 0, or `theta` is
/// not positive and finite.
pub fn rope_table(
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
) -> Result<(usize, usize, Vec<f32>, Vec<f32>)> {
    let tables = RotaryTables::new(head_dim, max_seq_len, theta)?;
    Ok((max_seq_len, head_dim / 2, tables.cos().to_vec(), tables.sin().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_model_config::Preset;

    #[test]
    fn a_preset_reports_its_exact_parameter_count() {
        let config = Preset::Smoke.default_config();
        let report = config_report(&config).unwrap();
        assert_eq!(report.params.total(), 117_056);
        assert_eq!(report.head_dim, config.d_model / config.n_heads);
    }

    #[test]
    fn an_invalid_configuration_reports_the_reason() {
        let mut config = Preset::Smoke.default_config();
        config.n_heads = 3;
        assert!(config_report(&config).is_err());
    }

    #[test]
    fn the_rope_table_starts_unrotated_and_covers_every_position() {
        let (positions, half, cos, sin) = rope_table(8, 5, 10_000.0).unwrap();
        assert_eq!((positions, half, cos.len(), sin.len()), (5, 4, 20, 20));
        assert_eq!(&cos[..4], &[1.0; 4]);
        assert_eq!(&sin[..4], &[0.0; 4]);
        assert!(rope_table(7, 5, 10_000.0).is_err());
    }
}
