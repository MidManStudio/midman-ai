// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-model-config.md, section "lib.rs"
// ============================================================================
//! midman-model-config: the architecture configuration of a MidMan model.
//!
//! [`ModelConfig`] describes a decoder-only transformer, validates it, counts its
//! parameters exactly, and reads and writes a strict TOML-compatible text format.
//! [`Preset`] names the standard sizes. See `docs/midman-model-config.md`.
//!
//! ```
//! use midman_model_config::{ModelConfig, Preset};
//!
//! let config = Preset::Small.default_config();
//! assert_eq!(config.param_count().unwrap(), 22_909_824);
//!
//! // The text form round-trips exactly.
//! let text = config.to_toml_string();
//! assert_eq!(ModelConfig::from_toml_str(&text).unwrap(), config);
//! ```

mod config;
mod presets;
mod text;

pub use config::{ModelConfig, ParamBreakdown, CONFIG_SCHEMA_VERSION};
pub use presets::Preset;
