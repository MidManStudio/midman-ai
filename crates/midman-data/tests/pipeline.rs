// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "tests/pipeline.rs"
// ============================================================================
//! The parts together: manifest, license policy, scrubber and deduplication, the way a
//! corpus will pass through them.

use midman_data::{ExactDedup, LicensePolicy, Manifest, Scrubber, Source, REDACTION};

fn source(id: &str, license: &str, category: &str) -> Source {
    Source {
        id: id.to_string(),
        url: format!("https://github.com/{id}"),
        revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
        license: license.to_string(),
        fetched: "2026-10-20".to_string(),
        category: category.to_string(),
    }
}

/// A key-looking value built at run time, so that no line of this file looks like a credential.
fn key() -> String {
    ["gh", "p_"].concat() + "aB3dE5gH7jK9mN1pQ3sT5vX7zA9cE1gI3kM5"
}

#[test]
fn a_manifest_decides_which_sources_are_read() {
    let mut manifest = Manifest::new();
    manifest.push(source("rust-lang/rust", "MIT OR Apache-2.0", "rust")).unwrap();
    manifest.push(source("gnu/bash", "GPL-3.0-or-later", "shell")).unwrap();
    manifest.push(source("mozilla/gecko", "MPL-2.0", "code_other")).unwrap();
    manifest.push(source("someone/unlicensed", "NONE", "rust")).unwrap();

    let reloaded = Manifest::from_text(&manifest.to_text()).unwrap();
    assert_eq!(reloaded, manifest);

    let policy = LicensePolicy::permissive_and_weak_copyleft();
    let split = reloaded.partition(&policy);
    let used: Vec<&str> = split.accepted.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(used, ["rust-lang/rust", "mozilla/gecko"]);
    let refused: Vec<(&str, &str)> =
        split.rejected.iter().map(|r| (r.source.id.as_str(), r.reason.as_str())).collect();
    assert_eq!(refused.len(), 2);
    assert_eq!(refused[0].0, "gnu/bash");
    assert!(refused[0].1.contains("GPL-3.0-or-later"));
    assert!(refused[1].1.contains("no license"));
}

#[test]
fn scrubbing_before_deduplication_merges_documents_that_differ_only_by_a_secret() {
    let scrubber = Scrubber::new();
    let first = format!("export GITHUB_TOKEN={}\ncargo publish\n", key());
    let second = format!(
        "export GITHUB_TOKEN={}\ncargo publish\n",
        ["gh", "p_"].concat() + "Zy9xW8vU7tS6rQ5pO4nM3lK2jI1hG0fE9dC8"
    );
    let third = "cargo build --release\n".to_string();

    // Raw, the first two are different documents.
    let mut raw = ExactDedup::new();
    assert!(raw.insert(first.as_bytes()));
    assert!(raw.insert(second.as_bytes()));

    // Scrubbed, they are the same one.
    let mut dedup = ExactDedup::new();
    let mut kept = Vec::new();
    for document in [&first, &second, &third, &first] {
        let scrubbed = scrubber.scrub(document);
        if dedup.insert(scrubbed.text.as_bytes()) {
            kept.push(scrubbed.text);
        }
    }
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0], format!("export GITHUB_TOKEN={REDACTION}\ncargo publish\n"));
    assert_eq!(kept[1], third);
    assert_eq!((dedup.unique(), dedup.duplicates(), dedup.checked()), (2, 2, 4));
    assert!(kept.iter().all(|text| !text.contains(&key())));
}
