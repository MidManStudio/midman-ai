// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "manifest.rs"
// ============================================================================
//! The provenance manifest: where every source of training text came from.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use midman_foundation::{Error, Result};

use crate::license::{LicenseExpr, LicensePolicy, Verdict};

/// The version of the manifest file format this build reads and writes.
pub const MANIFEST_VERSION: u32 = 1;

const WHAT: &str = "data manifest";
const MAGIC: &str = "midman-data-manifest";

/// The keys of a record, in the order they are written and must be read.
const KEYS: [&str; 6] = ["id", "url", "revision", "license", "fetched", "category"];

/// One source of training text: a repository at a commit, or a set of documents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    /// A short unique name, for example `rust-lang/rust`. Letters, digits and `. _ / -`.
    pub id: String,
    /// Where it was fetched from, as `scheme://...`, with no spaces.
    pub url: String,
    /// The commit it was taken at: 40 lowercase hex digits (SHA-1) or 64 (SHA-256).
    pub revision: String,
    /// Its license as an SPDX expression, for example `MIT OR Apache-2.0`.
    pub license: String,
    /// The day it was fetched, `YYYY-MM-DD`.
    pub fetched: String,
    /// The kind of text it provides, for example `rust` or `shell`: lowercase letters,
    /// digits, `_` and `-`, up to 32 characters.
    pub category: String,
}

/// The sources that make up a training corpus, in the order they were added.
///
/// A manifest is checked as it is built: a [`Source`] that does not satisfy the rules in its
/// field documentation is refused, so a manifest on disk or in memory is always valid.
///
/// ```
/// use midman_data::{LicensePolicy, Manifest, Source};
///
/// let mut manifest = Manifest::new();
/// manifest.push(Source {
///     id: "example/project".into(),
///     url: "https://example.com/project".into(),
///     revision: "0123456789abcdef0123456789abcdef01234567".into(),
///     license: "MIT OR Apache-2.0".into(),
///     fetched: "2026-10-20".into(),
///     category: "rust".into(),
/// }).unwrap();
///
/// let policy = LicensePolicy::permissive_and_weak_copyleft();
/// assert_eq!(manifest.partition(&policy).accepted.len(), 1);
/// assert_eq!(Manifest::from_text(&manifest.to_text()).unwrap(), manifest);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Manifest {
    sources: Vec<Source>,
}

/// A source the policy refused, with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejected<'a> {
    /// The refused source.
    pub source: &'a Source,
    /// Why the license was refused.
    pub reason: String,
}

/// A manifest split by a license policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition<'a> {
    /// The sources whose license the policy accepts, in manifest order.
    pub accepted: Vec<&'a Source>,
    /// The sources it refuses, in manifest order.
    pub rejected: Vec<Rejected<'a>>,
}

impl Manifest {
    /// An empty manifest.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a source.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if a field breaks its rule (the message names the
    /// field and the id) or if the id is already in the manifest. Nothing is added then.
    pub fn push(&mut self, source: Source) -> Result<()> {
        self.check_new(&source).map_err(Error::invalid)?;
        self.sources.push(source);
        Ok(())
    }

    fn check_new(&self, source: &Source) -> std::result::Result<(), String> {
        validate(source)?;
        if self.get(&source.id).is_some() {
            return Err(format!("id `{}` is already in the manifest", source.id));
        }
        Ok(())
    }

    /// The sources, in the order they were added.
    pub fn sources(&self) -> &[Source] {
        &self.sources
    }

    /// The source with this id, if there is one.
    pub fn get(&self, id: &str) -> Option<&Source> {
        self.sources.iter().find(|source| source.id == id)
    }

    /// The number of sources.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Whether there are no sources.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Splits the sources by whether `policy` accepts their license.
    pub fn partition(&self, policy: &LicensePolicy) -> Partition<'_> {
        let mut accepted = Vec::new();
        let mut rejected = Vec::new();
        for source in &self.sources {
            // A manifest only holds licenses that parse, so the error arm is not expected;
            // it is handled as a refusal rather than a panic.
            match policy.check(&source.license) {
                Ok(Verdict::Accepted) => accepted.push(source),
                Ok(Verdict::Rejected(reason)) => rejected.push(Rejected { source, reason }),
                Err(err) => rejected.push(Rejected { source, reason: err.to_string() }),
            }
        }
        Partition { accepted, rejected }
    }

    /// The manifest as text: a header line, then one blank line and six `key: value` lines
    /// per source.
    pub fn to_text(&self) -> String {
        let mut out = format!("{MAGIC} {MANIFEST_VERSION}\n");
        for source in &self.sources {
            out.push('\n');
            let values = [
                &source.id,
                &source.url,
                &source.revision,
                &source.license,
                &source.fetched,
                &source.category,
            ];
            for (key, value) in KEYS.iter().zip(values) {
                out.push_str(key);
                out.push_str(": ");
                out.push_str(value);
                out.push('\n');
            }
        }
        out
    }

    /// Reads a manifest from its text form.
    ///
    /// # Errors
    ///
    /// Returns an error naming the line for a wrong header, a missing final newline, a
    /// carriage return, a blank line where none belongs, a missing, repeated, unknown or
    /// misordered key, an empty value, or a field that breaks its rule; and
    /// [`Error::UnsupportedVersion`] for a version other than 1.
    pub fn from_text(text: &str) -> Result<Manifest> {
        if let Some(n) = text.split('\n').position(|line| line.contains('\r')) {
            return Err(Error::parse(WHAT, n + 1, "carriage returns are not allowed"));
        }
        if text.is_empty() {
            return Err(Error::parse(WHAT, 0, "the file is empty"));
        }
        if !text.ends_with('\n') {
            return Err(Error::parse(WHAT, 0, "the file must end with a newline"));
        }
        let lines: Vec<&str> = text.lines().collect();

        let header = lines[0];
        let version =
            header.strip_prefix(MAGIC).and_then(|rest| rest.strip_prefix(' ')).ok_or_else(
                || Error::parse(WHAT, 1, format!("the first line must start with `{MAGIC} `")),
            )?;
        let version: u32 = version
            .parse()
            .map_err(|_| Error::parse(WHAT, 1, format!("`{version}` is not a version number")))?;
        if version != MANIFEST_VERSION {
            return Err(Error::UnsupportedVersion {
                what: WHAT.to_string(),
                found: version,
                supported: MANIFEST_VERSION,
            });
        }

        let mut manifest = Manifest::new();
        let mut at = 1;
        while at < lines.len() {
            if !lines[at].is_empty() {
                return Err(Error::parse(
                    WHAT,
                    at + 1,
                    "expected a blank line before the next source",
                ));
            }
            at += 1;
            let first_line = at + 1;
            let mut values: Vec<String> = Vec::with_capacity(KEYS.len());
            for key in KEYS {
                let line = lines.get(at).ok_or_else(|| {
                    Error::parse(WHAT, at + 1, format!("the file ends where `{key}:` was expected"))
                })?;
                let value =
                    line.strip_prefix(key).and_then(|rest| rest.strip_prefix(": ")).ok_or_else(
                        || Error::parse(WHAT, at + 1, format!("expected `{key}: <value>`")),
                    )?;
                if value.is_empty() {
                    return Err(Error::parse(WHAT, at + 1, format!("`{key}` has no value")));
                }
                values.push(value.to_string());
                at += 1;
            }
            let [id, url, revision, license, fetched, category]: [String; 6] = values
                .try_into()
                .map_err(|_| Error::parse(WHAT, first_line, "internal: wrong number of values"))?;
            let source = Source { id, url, revision, license, fetched, category };
            manifest
                .check_new(&source)
                .map_err(|message| Error::parse(WHAT, first_line, message))?;
            manifest.sources.push(source);
        }
        Ok(manifest)
    }

    /// Writes the manifest to `path` without ever leaving a half-written file there.
    ///
    /// The text goes to a sibling file named `<path>.tmp`, is flushed to disk, and is then
    /// renamed over `path`. The containing directory is not synced.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] with the path if the file cannot be written; the temporary
    /// file is removed.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut tmp_name = path.as_os_str().to_owned();
        tmp_name.push(".tmp");
        let tmp = PathBuf::from(tmp_name);
        let result = self.write_then_rename(&tmp, path);
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result.map_err(|err| Error::Io(format!("{}: {err}", path.display())))
    }

    fn write_then_rename(&self, tmp: &Path, path: &Path) -> std::io::Result<()> {
        let mut file = fs::File::create(tmp)?;
        file.write_all(self.to_text().as_bytes())?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        fs::rename(tmp, path)
    }

    /// Reads a manifest from a file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] with the path if the file cannot be read, or the error of
    /// [`Manifest::from_text`].
    pub fn load(path: impl AsRef<Path>) -> Result<Manifest> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|err| Error::Io(format!("{}: {err}", path.display())))?;
        Manifest::from_text(&text)
    }
}

/// Checks every field of a source against its rule.
fn validate(source: &Source) -> std::result::Result<(), String> {
    let values = [
        ("id", &source.id),
        ("url", &source.url),
        ("revision", &source.revision),
        ("license", &source.license),
        ("fetched", &source.fetched),
        ("category", &source.category),
    ];
    for (field, value) in values {
        if value.is_empty() {
            return Err(format!("`{field}` of `{}` is empty", source.id));
        }
        if value.chars().any(char::is_control) {
            return Err(format!("`{field}` of `{}` contains a control character", source.id));
        }
        if value.trim() != value {
            return Err(format!("`{field}` of `{}` has spaces at an end", source.id));
        }
    }
    let bad = |field: &str, rule: &str| Err(format!("`{field}` of `{}` {rule}", source.id));

    let id_ok = source.id.len() <= 128
        && source
            .id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-'));
    if !id_ok {
        return Err(format!(
            "id `{}` must be at most 128 characters of letters, digits and . _ / -",
            source.id
        ));
    }
    if !url_ok(&source.url) {
        return bad(
            "url",
            "must look like `scheme://host/...`, without spaces, in at most 512 bytes",
        );
    }
    let hex = source.revision.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'));
    if !(hex && (source.revision.len() == 40 || source.revision.len() == 64)) {
        return bad("revision", "must be a commit id: 40 or 64 lowercase hex digits");
    }
    if let Err(err) = LicenseExpr::parse(&source.license) {
        return Err(format!("`license` of `{}`: {err}", source.id));
    }
    if !date_ok(&source.fetched) {
        return bad("fetched", "must be a real date written YYYY-MM-DD, from 1970");
    }
    let category_ok = source.category.len() <= 32
        && source
            .category
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
    if !category_ok {
        return bad("category", "must be at most 32 characters of a-z, 0-9, _ and -");
    }
    Ok(())
}

fn url_ok(url: &str) -> bool {
    if url.len() > 512 || url.chars().any(|c| c.is_whitespace()) {
        return false;
    }
    match url.split_once("://") {
        Some((scheme, rest)) => {
            let mut chars = scheme.chars();
            chars.next().is_some_and(|c| c.is_ascii_lowercase())
                && chars.all(|c| {
                    c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '+' | '.' | '-')
                })
                && !rest.is_empty()
        }
        None => false,
    }
}

fn divisible(value: u32, by: u32) -> bool {
    (value / by) * by == value
}

fn date_ok(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    let number = |range: std::ops::Range<usize>| -> Option<u32> {
        let digits = &bytes[range];
        if digits.iter().all(u8::is_ascii_digit) {
            digits.iter().try_fold(0u32, |n, d| Some(n * 10 + u32::from(d - b'0')))
        } else {
            None
        }
    };
    let (Some(year), Some(month), Some(day)) = (number(0..4), number(5..7), number(8..10)) else {
        return false;
    };
    let leap = divisible(year, 4) && (!divisible(year, 100) || divisible(year, 400));
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    year >= 1970 && (1..=days).contains(&day)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A change that breaks one field of a source.
    type Change = Box<dyn Fn(&mut Source)>;

    fn source(id: &str) -> Source {
        Source {
            id: id.to_string(),
            url: format!("https://github.com/{id}"),
            revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            license: "MIT OR Apache-2.0".to_string(),
            fetched: "2026-10-20".to_string(),
            category: "rust".to_string(),
        }
    }

    fn manifest_of(ids: &[&str]) -> Manifest {
        let mut manifest = Manifest::new();
        for id in ids {
            manifest.push(source(id)).unwrap();
        }
        manifest
    }

    #[test]
    fn push_get_len() {
        let mut manifest = Manifest::new();
        assert!(manifest.is_empty());
        manifest.push(source("a/b")).unwrap();
        manifest.push(source("c/d")).unwrap();
        assert_eq!(manifest.len(), 2);
        assert_eq!(manifest.get("c/d").unwrap().id, "c/d");
        assert!(manifest.get("x/y").is_none());
        assert_eq!(manifest.sources()[0].id, "a/b");
    }

    #[test]
    fn a_duplicate_id_is_refused_and_adds_nothing() {
        let mut manifest = manifest_of(&["a/b"]);
        let err = manifest.push(source("a/b")).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
        assert!(err.to_string().contains("already"), "{err}");
        assert_eq!(manifest.len(), 1);
    }

    #[test]
    fn bad_fields_are_refused_by_name() {
        let cases: Vec<(&str, Change)> = vec![
            ("id", Box::new(|s| s.id = String::new())),
            ("id", Box::new(|s| s.id = "has space".into())),
            ("id", Box::new(|s| s.id = "caf\u{e9}".into())),
            ("id", Box::new(|s| s.id = "x".repeat(129))),
            ("url", Box::new(|s| s.url = "github.com/x/y".into())),
            ("url", Box::new(|s| s.url = "https://".into())),
            ("url", Box::new(|s| s.url = "https://a b".into())),
            ("url", Box::new(|s| s.url = "1http://x".into())),
            ("url", Box::new(|s| s.url = format!("https://{}", "a".repeat(512)))),
            ("url", Box::new(|s| s.url = format!("https://{}", "a".repeat(505)))),
            ("fetched", Box::new(|s| s.fetched = "2026-06-31".into())),
            ("fetched", Box::new(|s| s.fetched = "2026-09-31".into())),
            ("fetched", Box::new(|s| s.fetched = "2026-11-31".into())),
            ("revision", Box::new(|s| s.revision = "main".into())),
            (
                "revision",
                Box::new(|s| s.revision = "0123456789ABCDEF0123456789ABCDEF01234567".into()),
            ),
            ("revision", Box::new(|s| s.revision = "0123456789abcdef".into())),
            (
                "revision",
                Box::new(|s| s.revision = "g123456789abcdef0123456789abcdef01234567".into()),
            ),
            ("license", Box::new(|s| s.license = "MIT OR".into())),
            ("license", Box::new(|s| s.license = "MIT\nOR ISC".into())),
            ("fetched", Box::new(|s| s.fetched = "2026-13-01".into())),
            ("fetched", Box::new(|s| s.fetched = "2026-02-29".into())),
            ("fetched", Box::new(|s| s.fetched = "2026-04-31".into())),
            ("fetched", Box::new(|s| s.fetched = "2026-00-10".into())),
            ("fetched", Box::new(|s| s.fetched = "2026-10-00".into())),
            ("fetched", Box::new(|s| s.fetched = "1969-12-31".into())),
            ("fetched", Box::new(|s| s.fetched = "2026-1-5".into())),
            ("fetched", Box::new(|s| s.fetched = "20261020".into())),
            ("fetched", Box::new(|s| s.fetched = "2026/10/20".into())),
            ("fetched", Box::new(|s| s.fetched = "+026-10-20".into())),
            ("category", Box::new(|s| s.category = "Rust".into())),
            ("category", Box::new(|s| s.category = "a b".into())),
            ("category", Box::new(|s| s.category = "x".repeat(33))),
            ("category", Box::new(|s| s.category = String::new())),
            ("category", Box::new(|s| s.category = " rust".into())),
            ("url", Box::new(|s| s.url = "https://x\ty".into())),
        ];
        for (field, change) in cases {
            let mut s = source("a/b");
            change(&mut s);
            let err = Manifest::new().push(s.clone()).unwrap_err();
            assert!(matches!(err, Error::InvalidArgument(_)), "{s:?}");
            let message = err.to_string();
            let named = if field == "id" {
                message.contains("id")
            } else {
                message.contains(&format!("`{field}`"))
            };
            assert!(named, "{field}: {message}");
        }
    }

    #[test]
    fn good_edge_values_are_accepted() {
        let mut s = source("a");
        s.revision = "f".repeat(64);
        s.fetched = "2024-02-29".to_string(); // a leap day
        s.category = "x".repeat(32);
        s.url = "file:///data/terminal-sessions".to_string();
        s.id = "x".repeat(128);
        s.license = "LicenseRef-in-house".to_string();
        Manifest::new().push(s).unwrap();
        // A URL of exactly 512 bytes.
        let mut s = source("a");
        s.url = format!("https://{}", "a".repeat(504));
        assert_eq!(s.url.len(), 512);
        Manifest::new().push(s).unwrap();
        for date in [
            "2000-02-29",
            "2400-02-29",
            "1970-01-01",
            "9999-12-31",
            "2026-04-30",
            "2026-06-30",
            "2026-09-30",
            "2026-11-30",
            "2026-12-31",
            "2026-01-31",
        ] {
            let mut s = source("a");
            s.fetched = date.to_string();
            assert!(Manifest::new().push(s).is_ok(), "{date}");
        }
        for date in ["1900-02-29", "2100-02-29", "2023-02-29"] {
            let mut s = source("a");
            s.fetched = date.to_string();
            assert!(Manifest::new().push(s).is_err(), "{date}");
        }
    }

    #[test]
    fn text_round_trips_in_order() {
        let manifest = manifest_of(&["rust-lang/rust", "a/b", "z.z/y_y-1"]);
        let text = manifest.to_text();
        assert!(text.starts_with("midman-data-manifest 1\n\nid: rust-lang/rust\nurl: "));
        assert!(text.ends_with("category: rust\n"));
        assert_eq!(Manifest::from_text(&text).unwrap(), manifest);
        assert_eq!(Manifest::from_text(&text).unwrap().to_text(), text);
    }

    #[test]
    fn the_empty_manifest_is_the_header_alone() {
        let text = Manifest::new().to_text();
        assert_eq!(text, "midman-data-manifest 1\n");
        assert!(Manifest::from_text(&text).unwrap().is_empty());
    }

    #[test]
    fn the_text_has_exactly_this_shape() {
        let text = manifest_of(&["a/b"]).to_text();
        assert_eq!(
            text,
            "midman-data-manifest 1\n\nid: a/b\nurl: https://github.com/a/b\n\
             revision: 0123456789abcdef0123456789abcdef01234567\nlicense: MIT OR Apache-2.0\n\
             fetched: 2026-10-20\ncategory: rust\n"
        );
    }

    fn reject(text: &str) -> Error {
        Manifest::from_text(text).unwrap_err()
    }

    #[test]
    fn reader_refuses_a_wrong_frame() {
        let good = manifest_of(&["a/b"]).to_text();
        assert!(matches!(reject(""), Error::Parse { .. }));
        assert!(matches!(reject("\n"), Error::Parse { line: 1, .. }));
        assert!(matches!(reject("midman-data-manifest\n"), Error::Parse { line: 1, .. }));
        assert!(matches!(reject("midman-data-manifest x\n"), Error::Parse { line: 1, .. }));
        assert!(matches!(reject("other 1\n"), Error::Parse { line: 1, .. }));
        assert!(matches!(
            reject("midman-data-manifest 2\n"),
            Error::UnsupportedVersion { found: 2, supported: 1, .. }
        ));
        assert!(matches!(reject("midman-data-manifest 0\n"), Error::UnsupportedVersion { .. }));
        // No final newline.
        assert!(reject(good.trim_end_matches('\n')).to_string().contains("newline"));
        // A carriage return anywhere, with the line it is on.
        let crlf = good.replace('\n', "\r\n");
        assert!(matches!(reject(&crlf), Error::Parse { line: 1, .. }));
        let one_cr = good.replacen("url: ", "url:\r ", 1);
        assert!(matches!(reject(&one_cr), Error::Parse { line: 4, .. }));
        // A blank line at the end, two blank lines between sources, text where a blank
        // line belongs.
        assert!(matches!(reject(&format!("{good}\n")), Error::Parse { .. }));
        let two =
            manifest_of(&["a/b", "c/d"]).to_text().replacen("\n\nid: c/d", "\n\n\nid: c/d", 1);
        assert!(matches!(reject(&two), Error::Parse { line: 10, .. }), "{}", reject(&two));
        let glued = good.replacen("\n\nid:", "\nid:", 1);
        assert!(matches!(reject(&glued), Error::Parse { line: 2, .. }));
    }

    #[test]
    fn reader_refuses_wrong_records_and_names_the_line() {
        let good = manifest_of(&["a/b"]).to_text();
        // A key missing, repeated, renamed or out of order.
        let missing = good.replacen("category: rust\n", "", 1);
        assert!(reject(&missing).to_string().contains("category"), "{}", reject(&missing));
        let repeated =
            good.replacen("fetched: 2026-10-20\n", "fetched: 2026-10-20\nfetched: 2026-10-21\n", 1);
        assert!(matches!(reject(&repeated), Error::Parse { line: 8, .. }));
        let renamed = good.replacen("url:", "link:", 1);
        assert!(matches!(reject(&renamed), Error::Parse { line: 4, .. }));
        let swapped = good.replacen(
            "id: a/b\nurl: https://github.com/a/b\n",
            "url: https://github.com/a/b\nid: a/b\n",
            1,
        );
        assert!(matches!(reject(&swapped), Error::Parse { line: 3, .. }));
        // No space after the colon, two spaces, an empty value.
        let tight = good.replacen("id: a/b", "id:a/b", 1);
        assert!(matches!(reject(&tight), Error::Parse { line: 3, .. }));
        let wide = good.replacen("id: a/b", "id:  a/b", 1);
        assert!(reject(&wide).to_string().contains("spaces"), "{}", reject(&wide));
        let empty = good.replacen("id: a/b", "id: ", 1);
        assert!(reject(&empty).to_string().contains("no value"), "{}", reject(&empty));
        // A bad field value is reported at the first line of its record.
        let bad_date = good.replacen("2026-10-20", "2026-02-30", 1);
        assert!(matches!(reject(&bad_date), Error::Parse { line: 3, .. }));
        assert!(reject(&bad_date).to_string().contains("fetched"));
        // A duplicate id is reported at the second record.
        let twice = manifest_of(&["a/b", "c/d"]).to_text().replacen("id: c/d", "id: a/b", 1);
        assert!(matches!(reject(&twice), Error::Parse { line: 10, .. }), "{}", reject(&twice));
        // Trailing junk after a complete record.
        assert!(matches!(reject(&format!("{good}junk\n")), Error::Parse { .. }));
    }

    #[test]
    fn partition_splits_by_policy_and_keeps_order() {
        let mut manifest = Manifest::new();
        for (id, license) in [
            ("a/mit", "MIT"),
            ("b/gpl", "GPL-3.0-only"),
            ("c/mpl", "MPL-2.0"),
            ("d/none", "NOASSERTION"),
            ("e/dual", "GPL-2.0-only OR Apache-2.0"),
            ("f/both", "MIT AND GPL-2.0-only"),
        ] {
            let mut s = source(id);
            s.license = license.to_string();
            manifest.push(s).unwrap();
        }
        let policy = LicensePolicy::permissive_and_weak_copyleft();
        let split = manifest.partition(&policy);
        let accepted: Vec<&str> = split.accepted.iter().map(|s| s.id.as_str()).collect();
        let rejected: Vec<&str> = split.rejected.iter().map(|r| r.source.id.as_str()).collect();
        assert_eq!(accepted, ["a/mit", "c/mpl", "e/dual"]);
        assert_eq!(rejected, ["b/gpl", "d/none", "f/both"]);
        assert!(split.rejected[0].reason.contains("GPL-3.0-only"));
        assert!(split.rejected[1].reason.contains("no license"));
        assert!(split.rejected[2].reason.contains("GPL-2.0-only"));
    }

    #[test]
    fn in_house_sources_pass_only_when_the_policy_allows_the_reference() {
        let mut manifest = Manifest::new();
        let mut s = source("midman/terminal");
        s.license = "LicenseRef-in-house".to_string();
        s.url = "file:///data/terminal".to_string();
        manifest.push(s).unwrap();
        let strict = LicensePolicy::permissive_and_weak_copyleft();
        assert_eq!(manifest.partition(&strict).rejected.len(), 1);
        let open = strict.allow("LicenseRef-in-house");
        assert_eq!(manifest.partition(&open).accepted.len(), 1);
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("midman-data-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn save_and_load_round_trip_and_replace_atomically() {
        let dir = temp_dir("manifest-save");
        let path = dir.join("manifest.txt");
        let first = manifest_of(&["a/b"]);
        first.save(&path).unwrap();
        assert_eq!(Manifest::load(&path).unwrap(), first);
        let second = manifest_of(&["a/b", "c/d"]);
        second.save(&path).unwrap();
        assert_eq!(Manifest::load(&path).unwrap(), second);
        assert!(!dir.join("manifest.txt.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn io_errors_name_the_path_and_leave_no_temporary_file() {
        let dir = temp_dir("manifest-io");
        let missing = dir.join("nope").join("manifest.txt");
        let err = manifest_of(&["a/b"]).save(missing).unwrap_err();
        assert!(matches!(err, Error::Io(_)));
        assert!(err.to_string().contains("nope"), "{err}");
        let err = Manifest::load(dir.join("absent.txt")).unwrap_err();
        assert!(matches!(err, Error::Io(_)));
        assert!(err.to_string().contains("absent.txt"), "{err}");
        // A file that is not UTF-8 is an I/O error with the path, not a panic.
        let bad = dir.join("bad.txt");
        fs::write(&bad, [0xff, 0xfe, b'\n']).unwrap();
        assert!(matches!(Manifest::load(&bad).unwrap_err(), Error::Io(_)));
        fs::remove_dir_all(&dir).unwrap();
    }
}
