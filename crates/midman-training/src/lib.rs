// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-training.md, section "lib.rs"
// ============================================================================
//! midman-training: the training loop, evaluation, and saving and resuming a run.
//!
//! A [`Trainer`] owns the model, the optimizer and the batcher. Each [`Trainer::step`] takes
//! the next batch, runs the forward pass and the loss, goes backward, clips the gradient,
//! asks the schedule for a learning rate and updates the weights. Everything a run depends
//! on is in the model's weights, the optimizer's moments and a position in the data, and
//! [`Trainer::save_state`] and [`Trainer::resume`] keep exactly that in one file, so that a
//! run stopped after step `k` and resumed gives, bit for bit, what a run that never stopped
//! gives. [`Trainer::export_weights`] writes the weights alone as an ordinary checkpoint.
//!
//! ```
//! use midman_data::{Sequences, TokenShard};
//! use midman_foundation::Rng;
//! use midman_model::MidManModel;
//! use midman_model_config::ModelConfig;
//! use midman_training::{TrainConfig, Trainer};
//!
//! # fn main() -> midman_foundation::Result<()> {
//! let model_config = ModelConfig {
//!     vocab_size: 12, d_model: 16, n_layers: 1, n_heads: 2, n_kv_heads: 1, d_ff: 32,
//!     max_seq_len: 8, rope_theta: 10_000.0, norm_eps: 1e-5, tie_embeddings: true,
//! };
//! // A repeating pattern is easy to learn.
//! let tokens: Vec<u32> = (0..400).map(|i| (i % 5) as u32).collect();
//! let data = Sequences::new(vec![TokenShard::new(12, 11, tokens)?], 8)?;
//! let model = MidManModel::new(&model_config, &mut Rng::seed_from_u64(1))?;
//!
//! let mut config = TrainConfig::new(4, 8, 30);
//! config.max_lr = 1e-2;
//! let mut trainer = Trainer::new(model, config, data)?;
//! let first = trainer.step()?.loss;
//! let mut last = first;
//! while !trainer.done() {
//!     last = trainer.step()?.loss;
//! }
//! assert!(last < first);
//! # Ok(())
//! # }
//! ```

mod config;
mod state;
mod trainer;

pub use config::TrainConfig;
pub use state::{load_state, progress_of, state_to_file, Progress, STATE_FORMAT, STATE_VERSION};
pub use trainer::{EvalReport, StepReport, Trainer};
