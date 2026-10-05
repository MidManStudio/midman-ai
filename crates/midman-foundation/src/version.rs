// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-foundation.md, section "version.rs"
// ============================================================================
//! Version information and format-version checks.

use crate::error::{Error, Result};

/// The version of the MidMan workspace crates, taken from this crate's manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Checks that an input's format version is the one this build reads.
///
/// There are no migrations yet, so anything other than an exact match is an
/// [`Error::UnsupportedVersion`]. `what` names the format in the error message.
pub fn check_schema_version(what: &str, found: u32, supported: u32) -> Result<()> {
    if found == supported {
        Ok(())
    } else {
        Err(Error::UnsupportedVersion { what: what.to_string(), found, supported })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_versions_pass() {
        assert!(check_schema_version("model config", 1, 1).is_ok());
    }

    #[test]
    fn mismatched_versions_name_the_format() {
        let err = check_schema_version("model config", 2, 1).unwrap_err();
        assert_eq!(
            err,
            Error::UnsupportedVersion { what: "model config".to_string(), found: 2, supported: 1 }
        );
    }

    #[test]
    fn version_has_three_numeric_parts() {
        let parts: Vec<&str> = VERSION.split('.').collect();
        assert_eq!(parts.len(), 3, "unexpected version string {VERSION}");
        assert!(parts.iter().all(|p| p.parse::<u32>().is_ok()));
    }
}
