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
//! - [`Packer`] tokenizes documents into [`TokenShard`]s, files of token ids with an
//!   end-of-text token after every document.
//! - [`Sequences`] cuts shards into training windows and [`Batcher`] hands them out in a
//!   seeded, resumable order, in the shape `MidManModel::loss` takes.
//!
//! Quality filters, near-duplicate detection, weighted mixing of sources and
//! fill-in-the-middle are not built yet. See `docs/midman-data.md` for the design and the limits.

mod batch;
mod dedup;
mod hash;
mod license;
mod manifest;
mod pack;
mod scrub;
mod shard;

pub use batch::{Batch, Batcher, Sequences};
pub use dedup::ExactDedup;
pub use hash::{sha256, to_hex, Sha256};
pub use license::{LicenseExpr, LicensePolicy, Verdict, PERMISSIVE, WEAK_COPYLEFT};
pub use manifest::{Manifest, Partition, Rejected, Source, MANIFEST_VERSION};
pub use pack::Packer;
pub use scrub::{Finding, Scrubbed, Scrubber, REDACTION};
pub use shard::{TokenShard, SHARD_VERSION};
