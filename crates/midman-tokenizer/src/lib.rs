// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tokenizer.md, section "lib.rs"
// ============================================================================
//! midman-tokenizer: a byte-level BPE tokenizer, written from scratch.
//!
//! [`Trainer`] learns merges from text, [`Tokenizer`] encodes and decodes with them, and
//! [`PreTokenizerConfig`] fixes how text is cut into pieces before merging. Every byte
//! string encodes and decodes exactly, valid UTF-8 or not.
//!
//! See `docs/midman-tokenizer.md` for the design, the measurements behind the default
//! settings, and what is not built yet.

mod pretokenize;
mod special;
mod tokenizer;
mod trainer;

pub use pretokenize::{Pieces, PreTokenizerConfig, WhitespaceRule, WordPrefix, MAX_PIECE_BYTES};
pub use special::{
    default_special_tokens, COMMAND, END_OF_TEXT, EXIT_STATUS, FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX,
    OUTPUT, PAD, SPECIAL_TOKENS,
};
pub use tokenizer::{SpecialMode, Tokenizer, FIRST_MERGE_ID, FORMAT_VERSION};
pub use trainer::{TrainStats, Trainer, TrainerConfig};
