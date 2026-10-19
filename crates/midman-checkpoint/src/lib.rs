// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-checkpoint.md, section "lib.rs"
// ============================================================================
//! midman-checkpoint: saving and loading model weights.
//!
//! A checkpoint is a standard safetensors file, so other tools can read it. The crate
//! has its own reader and writer for the `F32` dtype ([`TensorFile`]) and helpers that
//! save and load a model's [`midman_tensor::Parameter`]s by name, strictly and all or
//! nothing ([`save_parameters`], [`load_parameters`]).
//!
//! Optimizer state, training progress and version migrations are not here yet; see
//! `docs/midman-checkpoint.md` for what exists and what does not.

mod json;
mod params;
mod safetensors;

pub use params::{
    check_header, load_parameters, parameters_to_file, save_parameters, CHECKPOINT_VERSION,
    FORMAT_KEY, FORMAT_NAME, VERSION_KEY,
};
pub use safetensors::{NamedTensor, TensorFile};
