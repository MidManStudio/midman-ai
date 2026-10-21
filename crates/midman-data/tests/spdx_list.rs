// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "tests/spdx_list.rs"
// ============================================================================
//! The license parser and the default policy against the official SPDX list.
//!
//! `fixtures/spdx_ids.txt` and `fixtures/spdx_exceptions.txt` are generated from the
//! `license-list-data` repository of SPDX (see `fixtures/generate_fixtures.py`).

use std::collections::BTreeSet;

use midman_data::{LicenseExpr, LicensePolicy, PERMISSIVE, WEAK_COPYLEFT};

struct Entry {
    id: &'static str,
    deprecated: bool,
}

fn licenses() -> Vec<Entry> {
    include_str!("fixtures/spdx_ids.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.split(' ');
            let id = fields.next().expect("an id");
            let deprecated = fields.next().expect("a flag") == "1";
            Entry { id, deprecated }
        })
        .collect()
}

fn exceptions() -> Vec<&'static str> {
    include_str!("fixtures/spdx_exceptions.txt").lines().filter(|l| !l.starts_with('#')).collect()
}

#[test]
fn the_fixtures_are_the_expected_size() {
    assert_eq!(licenses().len(), 740);
    assert_eq!(exceptions().len(), 86);
}

#[test]
fn every_official_identifier_and_exception_parses() {
    let exceptions = exceptions();
    for entry in licenses() {
        let alone = LicenseExpr::parse(entry.id).unwrap_or_else(|e| panic!("{}: {e}", entry.id));
        assert!(matches!(alone, LicenseExpr::License { .. }), "{}", entry.id);
        if !entry.id.ends_with('+') {
            LicenseExpr::parse(&format!("{}+", entry.id)).unwrap();
        }
        let with = format!("{} WITH {}", entry.id, exceptions[0]);
        LicenseExpr::parse(&with).unwrap_or_else(|e| panic!("{with}: {e}"));
        let both = format!("{} OR ({} AND {})", entry.id, entry.id, entry.id);
        LicenseExpr::parse(&both).unwrap();
    }
    for exception in &exceptions {
        let text = format!("MIT WITH {exception}");
        LicenseExpr::parse(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
    }
}

#[test]
fn the_default_list_holds_only_current_official_identifiers() {
    let official: BTreeSet<&str> =
        licenses().iter().filter(|e| !e.deprecated).map(|e| e.id).collect();
    for id in PERMISSIVE.iter().chain(WEAK_COPYLEFT) {
        assert!(official.contains(id), "{id} is not a current SPDX identifier");
    }
}

#[test]
fn the_default_policy_accepts_exactly_the_listed_licenses_and_the_old_spellings_of_the_lgpl() {
    let policy = LicensePolicy::permissive_and_weak_copyleft();
    let accepted: BTreeSet<&str> = licenses()
        .iter()
        .map(|e| e.id)
        .filter(|id| policy.check(id).expect("official ids parse").is_accepted())
        .collect();
    let mut expected: BTreeSet<&str> = PERMISSIVE.iter().chain(WEAK_COPYLEFT).copied().collect();
    // The deprecated spellings of the three LGPL versions: `LGPL-2.1` is `LGPL-2.1-only` and
    // `LGPL-2.1+` is `LGPL-2.1-or-later`. They are real entries of the official list.
    expected.extend(["LGPL-2.0", "LGPL-2.0+", "LGPL-2.1", "LGPL-2.1+", "LGPL-3.0", "LGPL-3.0+"]);
    let extra: Vec<_> = accepted.difference(&expected).collect();
    let missing: Vec<_> = expected.difference(&accepted).collect();
    assert!(
        extra.is_empty() && missing.is_empty(),
        "accepted but not expected: {extra:?}; expected but not accepted: {missing:?}"
    );
}

#[test]
fn no_strong_copyleft_or_restricted_family_is_accepted() {
    let policy = LicensePolicy::permissive_and_weak_copyleft();
    let families = [
        "GPL",
        "AGPL",
        "GFDL",
        "SSPL",
        "EUPL",
        "OSL",
        "CC-BY-NC",
        "CC-BY-ND",
        "BUSL",
        "Sleepycat",
        "CPAL",
        "RPL",
        "Watcom",
        "APSL",
    ];
    for entry in licenses() {
        if families.iter().any(|f| entry.id.starts_with(f)) {
            assert!(!policy.check(entry.id).unwrap().is_accepted(), "{}", entry.id);
        }
    }
}

#[test]
fn an_exception_changes_nothing_for_any_official_license() {
    // `X WITH E` is accepted exactly when `X` is, for every license and a spread of exceptions.
    let policy = LicensePolicy::permissive_and_weak_copyleft();
    let exceptions = exceptions();
    for entry in licenses() {
        let plain = policy.check(entry.id).unwrap().is_accepted();
        for exception in exceptions.iter().step_by(17) {
            let with = policy.check(&format!("{} WITH {exception}", entry.id)).unwrap();
            assert_eq!(with.is_accepted(), plain, "{} WITH {exception}", entry.id);
        }
    }
}
