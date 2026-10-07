// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-model.md, section "lib.rs"
// ============================================================================
//! midman-model: the MidMan decoder-only language model.
//!
//! [`MidManModel`] is built from a [`ModelConfig`] and turns token ids into
//! next-token logits. Construction checks that the parameters it created add up
//! to exactly what [`ModelConfig::param_count`] says.
//!
//! ```
//! use midman_foundation::Rng;
//! use midman_model::MidManModel;
//! use midman_model_config::Preset;
//!
//! # fn main() -> midman_foundation::Result<()> {
//! let config = Preset::Smoke.default_config();
//! let model = MidManModel::new(&config, &mut Rng::seed_from_u64(0))?;
//! assert_eq!(model.num_parameters() as u64, config.param_count()?);
//!
//! // Two sequences of three tokens each.
//! let logits = model.forward(&[1, 2, 3, 4, 5, 6], 2, 3)?;
//! assert_eq!(logits.shape(), &[2, 3, config.vocab_size]);
//! # Ok(())
//! # }
//! ```
//!
//! See `docs/midman-model.md` for the layout and the initialization.
//!
//! [`ModelConfig`]: midman_model_config::ModelConfig
//! [`ModelConfig::param_count`]: midman_model_config::ModelConfig::param_count

mod model;

pub use model::{MidManModel, DEFAULT_INIT_STD};
