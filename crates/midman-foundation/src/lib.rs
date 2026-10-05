// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-foundation.md, section "lib.rs"
// ============================================================================
//! midman-foundation: shared errors, format versioning and the deterministic
//! random number generator used across MidMan.
//!
//! Everything here is standard-library only, so every other crate can depend on
//! it without pulling in anything else. See `docs/midman-foundation.md`.
//!
//! ```
//! use midman_foundation::Rng;
//!
//! // The same seed always gives the same stream.
//! let mut a = Rng::seed_from_u64(42);
//! let mut b = Rng::seed_from_u64(42);
//! assert_eq!(a.next_u64(), b.next_u64());
//! ```

mod error;
mod rng;
mod version;

pub use error::{Error, Result};
pub use rng::Rng;
pub use version::{check_schema_version, VERSION};
