// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-api.md, section "lib.rs"
// ============================================================================
// This crate exports a C ABI (src/ffi.rs), which is what the browser build and any
// native C or C++ caller link against. Exported functions that take pointers, and
// the memory helpers behind them, need `unsafe`, so the workspace's
// `unsafe_code = "deny"` is opted back out of here. Every `unsafe` block carries a
// `// SAFETY:` comment, and every `unsafe fn` documents its contract.
#![allow(unsafe_code)]
//! midman-api: interfaces for applications that embed MidMan.
//!
//! [`Lab`] trains a model on a fixed task from the outside, so a caller can watch
//! the loss, the gradients and the attention maps as it learns. [`config_report`]
//! and [`rope_table`] answer questions that need no model. [`ffi`] wraps all of it
//! in a C ABI with opaque handles, which is how the web lab runs the real Rust
//! code in the browser.
//!
//! ```
//! use midman_api::{Hyper, Lab, LabConfig};
//! use midman_model_config::Preset;
//!
//! # fn main() -> midman_foundation::Result<()> {
//! let mut lab = Lab::new(LabConfig {
//!     model: Preset::Smoke.default_config(),
//!     seed: 1,
//!     batch: 2,
//!     seq: 8,
//!     hyper: Hyper::default(),
//! })?;
//! let report = lab.step(3);
//! assert_eq!(report.records.len(), 3);
//! assert!(report.records[0].loss > 5.0); // near ln(256) before it has learned anything
//! # Ok(())
//! # }
//! ```
//!
//! See `docs/midman-api.md`.

pub mod ffi;
mod json;
mod lab;
mod report;

pub use lab::{
    AttentionMaps, Eval, Hyper, Lab, LabConfig, ParamStat, StepRecord, StepReport, MAX_PARAMS,
    MAX_TOKENS,
};
pub use report::{config_report, rope_table, ConfigReport};
