// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "lib.rs"
// ============================================================================
//! midman-data: what happens to training text before it is tokenized.
//!
//! - [`LicensePolicy`] decides from an SPDX license expression whether a source may be used.
//! - [`Manifest`] records where every source came from: URL, commit, license, date.
//! - [`Scrubber`] finds secrets in text and replaces them.
//! - [`ExactDedup`] drops documents whose bytes were already seen, using [`sha256`].
//!
//! Packing, quality filters, near-duplicate detection and sampling are not built yet. See
//! `docs/midman-data.md` for the design and the limits.

mod dedup;
mod hash;
mod license;
mod manifest;
mod scrub;

pub use dedup::ExactDedup;
pub use hash::{sha256, to_hex, Sha256};
pub use license::{LicenseExpr, LicensePolicy, Verdict, PERMISSIVE, WEAK_COPYLEFT};
pub use manifest::{Manifest, Partition, Rejected, Source, MANIFEST_VERSION};
pub use scrub::{Finding, Scrubbed, Scrubber, REDACTION};
