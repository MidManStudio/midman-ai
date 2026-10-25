// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "tests/ingest_manifest.rs"
// ============================================================================
//! A manifest written by `scripts/ingest.py`, read and judged by this crate.
//!
//! `fixtures/ingest_manifest.txt` is a copy of `scripts/fixtures/ingest-manifest.txt`: four
//! sources taken from a real run of `ingest.py crates` and `ingest.py manifest` (a repository
//! at a commit id, with the license crates.io lists for it). The script has its own writer of
//! the format, so this test keeps the two sides honest: the Rust reader must accept what the
//! script writes, and the Rust writer must give the same bytes back.

use midman_data::{LicensePolicy, Manifest};

const WRITTEN_BY_THE_SCRIPT: &str = include_str!("fixtures/ingest_manifest.txt");

#[test]
fn the_reader_accepts_what_the_script_writes() {
    let manifest = Manifest::from_text(WRITTEN_BY_THE_SCRIPT).expect("a valid manifest");
    let ids: Vec<&str> = manifest.sources().iter().map(|s| s.id.as_str()).collect();
    assert_eq!(
        ids,
        ["rust-lang/hashbrown", "dtolnay/syn", "BurntSushi/memchr", "dtolnay/unicode-ident"]
    );
    for source in manifest.sources() {
        assert_eq!(source.category, "rust");
        assert_eq!(source.fetched, "2026-10-25");
        assert_eq!(source.revision.len(), 40);
        assert!(source.url.starts_with("https://github.com/"));
    }
}

#[test]
fn the_writer_gives_the_same_bytes_back() {
    let manifest = Manifest::from_text(WRITTEN_BY_THE_SCRIPT).expect("a valid manifest");
    assert_eq!(manifest.to_text(), WRITTEN_BY_THE_SCRIPT);
}

#[test]
fn the_policy_accepts_the_listed_licenses_including_an_and_expression() {
    let manifest = Manifest::from_text(WRITTEN_BY_THE_SCRIPT).expect("a valid manifest");
    let policy = LicensePolicy::permissive_and_weak_copyleft();
    let partition = manifest.partition(&policy);
    assert_eq!(partition.accepted.len(), 4);
    assert!(partition.rejected.is_empty());
    let licenses: Vec<&str> = manifest.sources().iter().map(|s| s.license.as_str()).collect();
    assert!(licenses.contains(&"(MIT OR Apache-2.0) AND Unicode-3.0"));
    assert!(licenses.contains(&"Unlicense OR MIT"));
}

#[test]
fn a_changed_license_in_the_same_file_is_judged_by_the_policy() {
    let changed = WRITTEN_BY_THE_SCRIPT.replacen("Unlicense OR MIT", "GPL-3.0-only", 1);
    let manifest = Manifest::from_text(&changed).expect("still a valid manifest");
    let partition = manifest.partition(&LicensePolicy::permissive_and_weak_copyleft());
    assert_eq!(partition.accepted.len(), 3);
    assert_eq!(partition.rejected.len(), 1);
    assert_eq!(partition.rejected[0].source.id, "BurntSushi/memchr");
}
