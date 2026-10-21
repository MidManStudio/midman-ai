// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "scrub.rs"
// ============================================================================
//! Finding and replacing secrets in text.

use std::cmp::Reverse;

/// What a secret is replaced with.
pub const REDACTION: &str = "<redacted>";

/// The rule name of the keyword-and-entropy rule.
const GENERIC: &str = "generic-secret";

/// One secret found in a text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Finding {
    /// The rule that matched, for example `github-token`.
    pub rule: &'static str,
    /// Byte offset of the first byte of the secret in the original text.
    pub start: usize,
    /// Byte offset just past the last byte of the secret in the original text.
    pub end: usize,
}

/// A text with its secrets replaced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scrubbed {
    /// The text with each secret replaced by [`REDACTION`].
    pub text: String,
    /// What was replaced, with offsets into the original text.
    pub findings: Vec<Finding>,
}

/// Finds secrets in text and replaces them.
///
/// ```
/// use midman_data::{Scrubber, REDACTION};
///
/// let scrubber = Scrubber::new();
/// let password = "hunter2".repeat(2);
/// let line = format!("git clone https://builder:{password}@example.com/repo.git");
/// let scrubbed = scrubber.scrub(&line);
/// assert_eq!(scrubbed.text, format!("git clone https://builder:{REDACTION}@example.com/repo.git"));
/// assert_eq!(scrubbed.findings[0].rule, "url-password");
/// ```
///
/// The rules cover the shapes of common credentials, not every service: see the crate
/// documentation for the list. They favor finding too much over finding too little, because
/// a wrongly replaced string costs one token and a missed key costs more.
#[derive(Debug, Clone)]
pub struct Scrubber {
    generic: bool,
}

impl Default for Scrubber {
    fn default() -> Self {
        Self::new()
    }
}

impl Scrubber {
    /// A scrubber with every rule, including the keyword-and-entropy rule.
    pub fn new() -> Self {
        Scrubber { generic: true }
    }

    /// Turns off the keyword-and-entropy rule (`generic-secret`), the one most likely to
    /// replace something that is not a secret. The rules for known shapes stay on.
    #[must_use]
    pub fn without_generic(mut self) -> Self {
        self.generic = false;
        self
    }

    /// Finds the secrets in `text`, sorted by position and not overlapping.
    ///
    /// Where rules overlap, the one that starts first wins, then the longer match, then a
    /// rule for a known shape over the generic one.
    pub fn find(&self, text: &str) -> Vec<Finding> {
        let bytes = text.as_bytes();
        let mut found = Vec::new();
        scan_structured(bytes, &mut found);
        scan_identifiers(bytes, self.generic, &mut found);
        resolve(found)
    }

    /// Replaces every secret in `text` with [`REDACTION`].
    ///
    /// Scrubbing is idempotent: scrubbing the result finds nothing more.
    pub fn scrub(&self, text: &str) -> Scrubbed {
        let findings = self.find(text);
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        for finding in &findings {
            out.push_str(&text[last..finding.start]);
            out.push_str(REDACTION);
            last = finding.end;
        }
        out.push_str(&text[last..]);
        Scrubbed { text: out, findings }
    }
}

fn resolve(mut found: Vec<Finding>) -> Vec<Finding> {
    found.sort_by_key(|f| (f.start, Reverse(f.end - f.start), f.rule == GENERIC));
    let mut kept = Vec::with_capacity(found.len());
    let mut last_end = 0;
    for finding in found {
        if finding.start >= last_end {
            last_end = finding.end;
            kept.push(finding);
        }
    }
    kept
}

// ---------------------------------------------------------------------------
// Character classes
// ---------------------------------------------------------------------------

fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn alnum(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

fn url_safe(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

fn dashed(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-'
}

fn upper_base32(b: u8) -> bool {
    b.is_ascii_uppercase() || (b'2'..=b'7').contains(&b)
}

fn dotted_url_safe(b: u8) -> bool {
    url_safe(b) || b == b'.'
}

fn slashed(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'/'
}

fn is_base64(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=')
}

// ---------------------------------------------------------------------------
// Rules for tokens with a fixed prefix
// ---------------------------------------------------------------------------

/// A token that starts with a fixed prefix, then at least `min_body` characters of a class.
struct PrefixRule {
    rule: &'static str,
    prefix: &'static str,
    body: fn(u8) -> bool,
    min_body: usize,
}

const fn prefix_rule(
    rule: &'static str,
    prefix: &'static str,
    body: fn(u8) -> bool,
    min_body: usize,
) -> PrefixRule {
    PrefixRule { rule, prefix, body, min_body }
}

/// The rules for tokens with a prefix. A longer run than the minimum is replaced whole.
/// More specific prefixes come before the ones they share a start with (`sk-ant-` before `sk-`).
const PREFIX_RULES: &[PrefixRule] = &[
    prefix_rule("github-token", "ghp_", alnum, 36),
    prefix_rule("github-token", "gho_", alnum, 36),
    prefix_rule("github-token", "ghu_", alnum, 36),
    prefix_rule("github-token", "ghs_", alnum, 36),
    prefix_rule("github-token", "ghr_", alnum, 36),
    prefix_rule("github-token", "github_pat_", is_word, 50),
    prefix_rule("gitlab-token", "glpat-", url_safe, 20),
    prefix_rule("aws-access-key-id", "AKIA", upper_base32, 16),
    prefix_rule("aws-access-key-id", "ASIA", upper_base32, 16),
    prefix_rule("aws-access-key-id", "ABIA", upper_base32, 16),
    prefix_rule("aws-access-key-id", "ACCA", upper_base32, 16),
    prefix_rule("slack-token", "xoxb-", dashed, 20),
    prefix_rule("slack-token", "xoxa-", dashed, 20),
    prefix_rule("slack-token", "xoxp-", dashed, 20),
    prefix_rule("slack-token", "xoxr-", dashed, 20),
    prefix_rule("slack-token", "xoxs-", dashed, 20),
    prefix_rule("slack-token", "xapp-", dashed, 20),
    prefix_rule("slack-webhook", "https://hooks.slack.com/services/", slashed, 20),
    prefix_rule("stripe-key", "sk_live_", alnum, 10),
    prefix_rule("stripe-key", "sk_test_", alnum, 10),
    prefix_rule("stripe-key", "rk_live_", alnum, 10),
    prefix_rule("stripe-key", "rk_test_", alnum, 10),
    prefix_rule("google-api-key", "AIza", url_safe, 35),
    prefix_rule("anthropic-key", "sk-ant-", url_safe, 20),
    prefix_rule("openai-key", "sk-proj-", url_safe, 20),
    prefix_rule("openai-key", "sk-svcacct-", url_safe, 20),
    prefix_rule("openai-key", "sk-admin-", url_safe, 20),
    prefix_rule("openai-key", "sk-", alnum, 40),
    prefix_rule("npm-token", "npm_", alnum, 36),
    prefix_rule("pypi-token", "pypi-AgEI", url_safe, 50),
    prefix_rule("sendgrid-key", "SG.", dotted_url_safe, 60),
    prefix_rule("huggingface-token", "hf_", alnum, 30),
];

/// Which bytes some prefix rule starts with. Most words start with none of them, and
/// checking this first avoids comparing every rule's prefix at every word.
const fn first_bytes() -> [bool; 256] {
    let mut table = [false; 256];
    let mut i = 0;
    while i < PREFIX_RULES.len() {
        table[PREFIX_RULES[i].prefix.as_bytes()[0] as usize] = true;
        i += 1;
    }
    table
}

static STARTS_A_PREFIX_RULE: [bool; 256] = first_bytes();

fn match_prefix_rule(bytes: &[u8], i: usize) -> Option<Finding> {
    if !STARTS_A_PREFIX_RULE[usize::from(bytes[i])] {
        return None;
    }
    let rest = &bytes[i..];
    for rule in PREFIX_RULES {
        let prefix = rule.prefix.as_bytes();
        if prefix[0] == rest[0] && rest.starts_with(prefix) {
            let body = rest[prefix.len()..].iter().take_while(|&&b| (rule.body)(b)).count();
            if body >= rule.min_body {
                return Some(Finding { rule: rule.rule, start: i, end: i + prefix.len() + body });
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Structured rules: JWT, private keys, passwords in URLs
// ---------------------------------------------------------------------------

/// The length of the run of base64url characters at the start of `bytes`.
fn url_safe_run(bytes: &[u8]) -> usize {
    bytes.iter().take_while(|&&b| url_safe(b)).count()
}

/// A JSON web token: `eyJ...` header, `.`, `eyJ...` payload, `.`, signature (possibly empty).
fn match_jwt(bytes: &[u8], i: usize) -> Option<Finding> {
    let rest = &bytes[i..];
    if !rest.starts_with(b"eyJ") {
        return None;
    }
    let header = url_safe_run(rest);
    if header < 17 || rest.get(header) != Some(&b'.') {
        return None;
    }
    let payload_start = header + 1;
    let payload_part = &rest[payload_start..];
    if !payload_part.starts_with(b"eyJ") {
        return None;
    }
    let payload = url_safe_run(payload_part);
    if payload < 17 || payload_part.get(payload) != Some(&b'.') {
        return None;
    }
    let signature_start = payload_start + payload + 1;
    let signature = url_safe_run(&rest[signature_start..]);
    Some(Finding { rule: "jwt", start: i, end: i + signature_start + signature })
}

/// Most bytes of a private key body that are searched for its end.
const MAX_KEY_BODY: usize = 16 * 1024;

fn is_key_body_byte(b: u8) -> bool {
    is_base64(b) || b.is_ascii_whitespace() || matches!(b, b'\\' | b':' | b',' | b'.' | b'-' | b'_')
}

/// A PEM private key: from `-----BEGIN ... PRIVATE KEY-----` through the matching
/// `-----END` line, including keys whose line breaks are written as the two characters `\n`.
///
/// The body must start with a run of at least 40 base64 characters (or a `Proc-Type:`
/// header) and hold at least 64 of them, so a mention of the header in documentation does
/// not swallow the text after it. A key whose END line is missing is replaced through the
/// end of its run of base64-looking text.
fn match_private_key(bytes: &[u8], i: usize) -> Option<Finding> {
    let rest = &bytes[i..];
    let begin = b"-----BEGIN ";
    if !rest.starts_with(begin) {
        return None;
    }
    let window = &rest[begin.len()..rest.len().min(begin.len() + 64)];
    let label_len = window.windows(5).position(|w| w == b"-----")?;
    let label = &window[..label_len];
    if !label.iter().all(|&b| b.is_ascii_uppercase() || b == b' ') {
        return None;
    }
    if !label.windows(11).any(|w| w == b"PRIVATE KEY") {
        return None;
    }
    let body_start = begin.len() + label_len + 5;
    let body = &rest[body_start..];

    let mut skip = 0;
    loop {
        match body.get(skip) {
            Some(b) if b.is_ascii_whitespace() => skip += 1,
            Some(b'\\') if matches!(body.get(skip + 1), Some(b'n' | b'r' | b't')) => skip += 2,
            _ => break,
        }
    }
    let first_run = body[skip..].iter().take_while(|&&b| is_base64(b)).count();
    if first_run < 40 && !body[skip..].starts_with(b"Proc-Type:") {
        return None;
    }

    let mut length = 0;
    let mut base64_count = 0;
    while length < body.len().min(MAX_KEY_BODY) {
        let b = body[length];
        if body[length..].starts_with(b"-----") || !is_key_body_byte(b) {
            break;
        }
        if is_base64(b) {
            base64_count += 1;
        }
        length += 1;
    }
    if base64_count < 64 {
        return None;
    }
    let mut end = body_start + length;
    let after = &rest[end..];
    if after.starts_with(b"-----END ") {
        let tail = &after[9..after.len().min(9 + 64)];
        if let Some(close) = tail.windows(5).position(|w| w == b"-----") {
            end += 9 + close + 5;
        }
    }
    Some(Finding { rule: "private-key", start: i, end: i + end })
}

fn userinfo_char(b: u8) -> bool {
    !(b.is_ascii_whitespace()
        || matches!(b, b'/' | b'@' | b'"' | b'\'' | b'<' | b'>' | b'`' | b'\\' | b'?' | b'#'))
}

/// The password in `scheme://user:<password>@host`; `colon_slash_slash` is the index of the
/// `:` of `://`. Only the password is returned. A user name may be empty (`redis://:<password>@host`).
/// Passwords that start like a template (`$`, `{`, `%`, `*`, `[`) are left alone.
fn match_url_password(bytes: &[u8], colon_slash_slash: usize) -> Option<Finding> {
    let i = colon_slash_slash;
    if i == 0
        || !(bytes[i - 1].is_ascii_alphanumeric() || matches!(bytes[i - 1], b'+' | b'.' | b'-'))
    {
        return None;
    }
    let user_start = i + 3;
    let user_len =
        bytes[user_start..].iter().take(64).take_while(|&&b| userinfo_char(b) && b != b':').count();
    let colon = user_start + user_len;
    if bytes.get(colon) != Some(&b':') {
        return None;
    }
    let password_start = colon + 1;
    let password_len =
        bytes[password_start..].iter().take(256).take_while(|&&b| userinfo_char(b)).count();
    if password_len == 0 || matches!(bytes[password_start], b'$' | b'{' | b'%' | b'*' | b'[') {
        return None;
    }
    let password_end = password_start + password_len;
    if bytes.get(password_end) != Some(&b'@') {
        return None;
    }
    match bytes.get(password_end + 1) {
        Some(&b) if b.is_ascii_alphanumeric() || b == b'[' => {}
        _ => return None,
    }
    Some(Finding { rule: "url-password", start: password_start, end: password_end })
}

fn scan_structured(bytes: &[u8], out: &mut Vec<Finding>) {
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let mut found = None;
        if i == 0 || !is_word(bytes[i - 1]) {
            found = match_prefix_rule(bytes, i).or_else(|| match_jwt(bytes, i));
        }
        if found.is_none() && b == b'-' {
            found = match_private_key(bytes, i);
        }
        if found.is_none() && b == b':' && bytes[i..].starts_with(b"://") {
            found = match_url_password(bytes, i);
        }
        match found {
            Some(finding) => {
                // A URL password starts after `i`; nothing between `i` and it starts a rule.
                i = finding.end.max(i + 1);
                out.push(finding);
            }
            None => i += 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Rules that start from an identifier: Authorization headers and secret assignments
// ---------------------------------------------------------------------------

/// Words that, inside an identifier, mark the value after it as probably a secret.
const KEYWORDS: &[&str] = &[
    "secret",
    "password",
    "passwd",
    "pwd",
    "token",
    "apikey",
    "api_key",
    "api-key",
    "access_key",
    "accesskey",
    "private_key",
    "privatekey",
];

/// Shortest and longest value the generic rule replaces.
const GENERIC_MIN: usize = 16;
const GENERIC_MAX: usize = 256;

/// Lowest Shannon entropy, in bits per character, of a value the generic rule replaces.
const GENERIC_MIN_ENTROPY: f64 = 3.5;

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

fn skip_blanks(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    i
}

fn contains_ignore_case(haystack: &[u8], needle: &str) -> bool {
    let needle = needle.as_bytes();
    haystack.len() >= needle.len()
        && haystack.windows(needle.len()).any(|w| w.eq_ignore_ascii_case(needle))
}

fn scan_identifiers(bytes: &[u8], generic: bool, out: &mut Vec<Finding>) {
    let mut i = 0;
    while i < bytes.len() {
        if !is_ident(bytes[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && is_ident(bytes[i]) {
            i += 1;
        }
        let mut ident = &bytes[start..i];
        while ident.first() == Some(&b'-') {
            ident = &ident[1..];
        }
        if ident.is_empty() || ident.len() > 64 {
            continue;
        }
        if contains_ignore_case(ident, "authorization") {
            if let Some(finding) = match_auth_header(bytes, i) {
                out.push(finding);
            }
        }
        if generic && KEYWORDS.iter().any(|k| contains_ignore_case(ident, k)) {
            if let Some(finding) = match_assignment(bytes, i) {
                out.push(finding);
            }
        }
    }
}

/// After the end of a key (`after_key`), skips an optional closing quote and blanks and
/// reads one of `=`, `==`, `===`, `:`, `:=`, `=>`. Returns the index after it, or `None` for
/// anything else, including `::`. A comparison is read like an assignment: a secret that
/// is compared against in code is written in the code.
fn skip_assignment_operator(bytes: &[u8], after_key: usize) -> Option<usize> {
    let mut p = after_key;
    if matches!(bytes.get(p), Some(b'"' | b'\'')) {
        p += 1;
    }
    p = skip_blanks(bytes, p);
    let length = match (bytes.get(p), bytes.get(p + 1)) {
        (Some(b':'), Some(b'=')) | (Some(b'='), Some(b'>')) => 2,
        (Some(b'='), Some(b'=')) => {
            if bytes.get(p + 2) == Some(&b'=') {
                3
            } else {
                2
            }
        }
        (Some(b':'), Some(b':')) => return None,
        (Some(b'='), _) | (Some(b':'), _) => 1,
        _ => return None,
    };
    Some(skip_blanks(bytes, p + length))
}

/// `Authorization: Bearer <token>` (also `Basic` and `Token`, with `=` or quotes). Only the
/// credential is replaced, and only if it is at least 16 characters of token text, so
/// `Bearer $TOKEN` and `Bearer {token}` stay.
fn match_auth_header(bytes: &[u8], after_key: usize) -> Option<Finding> {
    let mut p = skip_assignment_operator(bytes, after_key)?;
    if matches!(bytes.get(p), Some(b'"' | b'\'')) {
        p += 1;
    }
    let scheme = bytes[p..].iter().take_while(|b| b.is_ascii_alphabetic()).count();
    let name = &bytes[p..p + scheme];
    if !["bearer", "basic", "token"].iter().any(|s| name.eq_ignore_ascii_case(s.as_bytes())) {
        return None;
    }
    p += scheme;
    let blanks = skip_blanks(bytes, p);
    if blanks == p {
        return None;
    }
    let credential = bytes[blanks..]
        .iter()
        .take_while(|&&b| {
            b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'~' | b'+' | b'/' | b'=' | b'-')
        })
        .count();
    if credential < 16 {
        return None;
    }
    Some(Finding { rule: "auth-header", start: blanks, end: blanks + credential })
}

/// `key = value` where the key contains a keyword and the value looks random.
fn match_assignment(bytes: &[u8], after_key: usize) -> Option<Finding> {
    let p = skip_assignment_operator(bytes, after_key)?;
    let (start, end) = match bytes.get(p) {
        Some(&quote) if matches!(quote, b'"' | b'\'' | b'`') => {
            let start = p + 1;
            let length = bytes[start..]
                .iter()
                .take(GENERIC_MAX + 1)
                .take_while(|&&b| b != quote && !b.is_ascii_whitespace())
                .count();
            if bytes.get(start + length) != Some(&quote) {
                return None;
            }
            (start, start + length)
        }
        Some(_) => {
            let length = bytes[p..]
                .iter()
                .take(GENERIC_MAX + 1)
                .take_while(|&&b| is_base64(b) || b == b'_' || b == b'-')
                .count();
            let end = p + length;
            // A value that is the start of a call, a path or a field access, or that sits in
            // an argument, parameter or field list (`private_key: SomeType,`), is code.
            let closes_a_list = |b: Option<&u8>| matches!(b, Some(b',' | b')' | b']' | b'}'));
            if matches!(bytes.get(end), Some(b'(' | b'.' | b'[' | b':' | b'{'))
                || closes_a_list(bytes.get(skip_blanks(bytes, end)))
            {
                return None;
            }
            (p, end)
        }
        None => return None,
    };
    let value = &bytes[start..end];
    if looks_like_a_secret(value) {
        Some(Finding { rule: GENERIC, start, end })
    } else {
        None
    }
}

fn looks_like_a_secret(value: &[u8]) -> bool {
    if !(GENERIC_MIN..=GENERIC_MAX).contains(&value.len()) || !value.is_ascii() {
        return false;
    }
    if !value.iter().any(u8::is_ascii_digit) || !value.iter().any(u8::is_ascii_alphabetic) {
        return false;
    }
    // Paths, URLs and templates. A base64 secret can start with `/`, so that alone does not
    // make a path: a path also has one of `. - _ ~` or no capital letter.
    if matches!(value[0], b'.' | b'~') || contains_sub(value, b"://") {
        return false;
    }
    if value[0] == b'/' {
        let path_characters = value.iter().any(|b| matches!(b, b'.' | b'-' | b'_' | b'~'));
        if path_characters || !value.iter().any(u8::is_ascii_uppercase) {
            return false;
        }
    }
    if contains_sub(value, b"${") || contains_sub(value, b"{{") || contains_sub(value, b"<%") {
        return false;
    }
    // snake_case names such as `some_long_function_name_2`.
    let lower_snake =
        value.iter().all(|&b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
    if lower_snake && value.contains(&b'_') {
        return false;
    }
    entropy(value) >= GENERIC_MIN_ENTROPY
}

fn contains_sub(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Shannon entropy of the bytes, in bits per byte.
fn entropy(bytes: &[u8]) -> f64 {
    let mut counts = [0usize; 256];
    for &b in bytes {
        counts[usize::from(b)] += 1;
    }
    let total = bytes.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.log2()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic string of `n` characters from `alphabet`, with no pattern a person
    /// would call a word. Test secrets are built from pieces like this at run time so that
    /// no source line looks like a credential to a scanner.
    fn random_text(n: usize, alphabet: &str, seed: u64) -> String {
        let chars: Vec<char> = alphabet.chars().collect();
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                chars[(state >> 11) as usize % chars.len()]
            })
            .collect()
    }

    /// Five dashes. Test PEM text is assembled around this so that no line of this file holds a
    /// complete BEGIN or END line, which a secret scanner would report.
    const DASHES: &str = "-----";

    const ALNUM: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    const URL_SAFE: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";
    const BASE32: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

    fn scrubber() -> Scrubber {
        Scrubber::new()
    }

    fn rules(text: &str) -> Vec<&'static str> {
        scrubber().find(text).iter().map(|f| f.rule).collect()
    }

    fn only_finding(text: &str) -> Finding {
        let found = scrubber().find(text);
        assert_eq!(found.len(), 1, "{text:?} -> {found:?}");
        found[0]
    }

    fn spans(text: &str) -> Vec<&str> {
        scrubber().find(text).iter().map(|f| &text[f.start..f.end]).collect()
    }

    #[test]
    fn every_prefix_rule_finds_a_token_of_the_minimum_length() {
        for (n, rule) in PREFIX_RULES.iter().enumerate() {
            let alphabet: String =
                (0u8..=127).filter(|&b| (rule.body)(b)).map(char::from).collect();
            let body = random_text(rule.min_body, &alphabet, n as u64 + 1);
            let token = format!("{}{body}", rule.prefix);
            for text in
                [token.clone(), format!("key = \"{token}\" # x"), format!("export X={token}\nnext")]
            {
                let found = scrubber().find(&text);
                assert!(
                    found.iter().any(|f| f.rule == rule.rule && text[f.start..f.end] == token),
                    "{} {:?}: {text:?} -> {found:?}",
                    rule.rule,
                    rule.prefix
                );
            }
            // One character fewer is not a token.
            if rule.min_body > 0 {
                let short = format!("{}{}", rule.prefix, &body[..body.len() - 1]);
                let text = format!("x = {short} y");
                assert!(
                    !scrubber().find(&text).iter().any(|f| f.rule == rule.rule),
                    "{} {:?} too short: {text:?}",
                    rule.rule,
                    rule.prefix
                );
            }
        }
    }

    #[test]
    fn a_longer_run_is_replaced_whole() {
        let body = random_text(60, ALNUM, 3);
        let token = format!("{}{body}", ["gh", "p_"].concat());
        let text = format!("a {token} b");
        assert_eq!(spans(&text), [token.as_str()]);
        let key = format!("{}{}", ["AK", "IA"].concat(), random_text(24, BASE32, 4));
        assert_eq!(spans(&format!("id: {key}")), [key.as_str()]);
    }

    #[test]
    fn a_prefix_inside_a_word_is_not_a_token() {
        let body = random_text(40, ALNUM, 5);
        let prefix = ["gh", "p_"].concat();
        for before in ["x", "_", "7", "Z"] {
            let text = format!("{before}{prefix}{body}");
            assert!(rules(&text).is_empty(), "{text:?}");
        }
        for before in [" ", "\"", "=", ":", "(", "-", "/", "\n", "\u{e9}", ">"] {
            let text = format!("{before}{prefix}{body}");
            assert_eq!(rules(&text), ["github-token"], "{text:?}");
        }
        // Start of the text counts as a boundary.
        assert_eq!(rules(&format!("{prefix}{body}")), ["github-token"]);
    }

    #[test]
    fn specific_prefixes_win_over_the_ones_they_share_a_start_with() {
        let ant = format!("{}{}", ["sk", "-ant-"].concat(), random_text(60, URL_SAFE, 6));
        assert_eq!(only_finding(&ant).rule, "anthropic-key");
        let proj = format!("{}{}", ["sk", "-proj-"].concat(), random_text(60, URL_SAFE, 7));
        assert_eq!(only_finding(&proj).rule, "openai-key");
        let old = format!("{}{}", ["sk", "-"].concat(), random_text(48, ALNUM, 8));
        assert_eq!(only_finding(&old).rule, "openai-key");
        // `sk-` followed by a word is just text.
        assert!(rules("the sk-learn package and task-runner").is_empty());
    }

    #[test]
    fn a_slack_webhook_is_replaced_whole() {
        let url = format!(
            "{}{}",
            ["https://hooks.", "slack.com/services/"].concat(),
            [
                "T",
                &random_text(8, ALNUM, 9),
                "/B",
                &random_text(8, ALNUM, 10),
                "/",
                &random_text(24, ALNUM, 11)
            ]
            .concat()
        );
        let text = format!("curl -X POST {url} -d x");
        assert_eq!(spans(&text), [url.as_str()]);
        assert_eq!(only_finding(&text).rule, "slack-webhook");
    }

    #[test]
    fn jwt_needs_two_ey_parts_and_a_dot_after_each() {
        let header = format!("eyJ{}", random_text(30, URL_SAFE, 12));
        let payload = format!("eyJ{}", random_text(40, URL_SAFE, 13));
        let signature = random_text(43, URL_SAFE, 14);
        let jwt = format!("{header}.{payload}.{signature}");
        let text = format!("Cookie: session={jwt}; path=/");
        assert_eq!(spans(&text), [jwt.as_str()]);
        assert_eq!(only_finding(&text).rule, "jwt");
        // An empty signature (alg none) still counts.
        let unsigned = format!("{header}.{payload}.");
        assert_eq!(spans(&format!("t={unsigned} z")), [unsigned.as_str()]);
        // Two parts, a payload that does not start with eyJ, parts that are too short.
        assert!(rules(&format!("{header}.{payload}")).is_empty());
        assert!(
            rules(&format!("{header}.{}.{signature}", random_text(40, URL_SAFE, 15))).is_empty()
        );
        assert!(rules("eyJabc.eyJdef.ghi").is_empty());
    }

    fn pem(label: &str, lines: usize, seed: u64, newline: &str) -> String {
        let mut text = format!("{DASHES}BEGIN {label}{DASHES}{newline}");
        for n in 0..lines {
            let alphabet = format!("{ALNUM}+/");
            text.push_str(&random_text(64, &alphabet, seed + n as u64));
            text.push_str(newline);
        }
        text.push_str(&format!("{DASHES}END {label}{DASHES}"));
        text
    }

    #[test]
    fn a_private_key_is_replaced_from_begin_to_end() {
        for label in [
            "RSA PRIVATE KEY",
            "PRIVATE KEY",
            "EC PRIVATE KEY",
            "OPENSSH PRIVATE KEY",
            "ENCRYPTED PRIVATE KEY",
        ] {
            let key = pem(label, 12, 20, "\n");
            let text = format!("config:\n{key}\nafter\n");
            assert_eq!(spans(&text), [key.as_str()], "{label}");
            let scrubbed = scrubber().scrub(&text);
            assert_eq!(scrubbed.text, format!("config:\n{REDACTION}\nafter\n"));
        }
        // PGP private key blocks have a longer label.
        let key = pem("PGP PRIVATE KEY BLOCK", 6, 30, "\n");
        assert_eq!(spans(&key), [key.as_str()]);
    }

    #[test]
    fn a_private_key_written_with_escaped_line_breaks_is_found() {
        let key = pem("RSA PRIVATE KEY", 8, 40, "\\n");
        let text = format!("{{\"private_key\": \"{key}\\n\", \"client_email\": \"x\"}}");
        assert_eq!(spans(&text), [key.as_str()]);
        let crlf = pem("PRIVATE KEY", 5, 50, "\r\n");
        assert_eq!(spans(&crlf), [crlf.as_str()]);
    }

    #[test]
    fn a_pem_with_headers_is_found() {
        let alphabet = format!("{ALNUM}+/");
        let text = format!(
            "{DASHES}BEGIN RSA PRIVATE KEY{DASHES}\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC,{}\n\n{}\n{}\n{DASHES}END RSA PRIVATE KEY{DASHES}\n",
            random_text(32, "0123456789ABCDEF", 60),
            random_text(64, &alphabet, 61),
            random_text(64, &alphabet, 62)
        );
        let found = scrubber().find(&text);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            text[found[0].end..].trim().is_empty(),
            "stopped early: {:?}",
            &text[found[0].end..]
        );
    }

    #[test]
    fn a_key_without_its_end_line_is_replaced_through_its_body() {
        let mut text = format!("{DASHES}BEGIN PRIVATE KEY{DASHES}\n");
        for n in 0..4 {
            text.push_str(&random_text(64, ALNUM, 70 + n));
            text.push('\n');
        }
        text.push_str("{ \"next\": 1 }\n");
        let found = only_finding(&text);
        assert_eq!(found.rule, "private-key");
        assert!(text[found.end..].contains("next"), "{:?}", &text[found.end..]);
        assert!(!text[found.start..found.end].contains('{'));
    }

    #[test]
    fn a_mention_of_the_header_in_documentation_is_not_a_key() {
        let d = DASHES;
        for text in [
            format!("{d}BEGIN RSA PRIVATE KEY{d}"),
            format!("The file starts with {d}BEGIN PRIVATE KEY{d} and then base64."),
            format!(
                "{d}BEGIN RSA PRIVATE KEY{d}\nand put your key here, then run the command below to continue the setup\n{d}END RSA PRIVATE KEY{d}"
            ),
            format!(
                "{d}BEGIN CERTIFICATE{d}\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAx5h3RmR3yq7tXlq8vJ9i0b8c\n{d}END CERTIFICATE{d}"
            ),
            format!(
                "{d}BEGIN PUBLIC KEY{d}\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAx5h3RmR3yq7tXlq8vJ9i0b8cABCD\n{d}END PUBLIC KEY{d}"
            ),
            format!("{d}BEGIN PRIVATE KEY{d}\nshort\n{d}END PRIVATE KEY{d}"),
            format!("----BEGIN PRIVATE KEY{d}"),
        ] {
            assert!(rules(&text).is_empty(), "{text:?} -> {:?}", scrubber().find(&text));
        }
    }

    #[test]
    fn only_the_password_of_a_url_is_replaced() {
        let password = random_text(14, ALNUM, 80);
        let text = format!("git clone https://builder:{password}@github.com/org/repo.git");
        assert_eq!(spans(&text), [password.as_str()]);
        assert_eq!(only_finding(&text).rule, "url-password");
        let scrubbed = scrubber().scrub(&text);
        assert_eq!(
            scrubbed.text,
            format!("git clone https://builder:{REDACTION}@github.com/org/repo.git")
        );
        // An empty user name, other schemes, a password with punctuation.
        let text = format!("redis://:{password}@cache:6379/0");
        assert_eq!(spans(&text), [password.as_str()]);
        let text = format!("postgres://app:{}@db.internal:5432/app", "p%40ss!w0rd.x");
        assert_eq!(spans(&text), ["p%40ss!w0rd.x"]);
        let text = format!("amqp://guest:{}@localhost", "guest");
        assert_eq!(spans(&text), ["guest"]);
    }

    #[test]
    fn urls_without_a_password_and_template_passwords_are_left_alone() {
        for text in [
            "https://example.com:8080/path",
            "http://localhost:3000",
            "ssh://git@github.com:22/org/repo",
            "https://user@example.com/x",
            "https://user:@example.com/x",
            "postgres://user:${DB_PASSWORD}@host/db",
            "postgres://user:{password}@host/db",
            "postgres://user:%s@host/db",
            "postgres://user:****@host/db",
            "postgres://user:$PASS@host/db",
            "postgres://user:<password>@host/db",
            "ftp://user:pw@",
            "://x:y@z",
            "see https://docs.example.com/guide?a=b:c@d",
        ] {
            assert!(rules(text).is_empty(), "{text:?} -> {:?}", scrubber().find(text));
        }
    }

    #[test]
    fn authorization_headers_lose_only_the_credential() {
        let token = random_text(30, ALNUM, 90);
        for (text, scheme) in [
            (format!("curl -H \"Authorization: Bearer {token}\" https://api"), "Bearer"),
            (format!("Authorization: Basic {token}"), "Basic"),
            (format!("authorization: token {token}"), "token"),
            (format!("headers = {{'Authorization': 'Bearer {token}'}}"), "Bearer"),
            (format!("\"Authorization\" = \"bearer {token}\""), "bearer"),
        ] {
            assert_eq!(spans(&text), [token.as_str()], "{scheme}: {text}");
            assert_eq!(only_finding(&text).rule, "auth-header");
        }
        for text in [
            "Authorization: Bearer $TOKEN",
            "Authorization: Bearer {token}",
            "Authorization: Bearer <token>",
            "Authorization: Bearer short",
            "Authorization: Digest username=\"x\"",
            "Authorization: Bearer",
            "Authorization = authorization_header",
            "authorization_url = \"https://example.com/oauth/authorize\"",
        ] {
            assert!(rules(text).is_empty(), "{text:?} -> {:?}", scrubber().find(text));
        }
    }

    #[test]
    fn secret_assignments_are_found_in_common_syntaxes() {
        let value = random_text(32, ALNUM, 100);
        let cases = [
            format!("API_KEY={value}"),
            format!("export SECRET_TOKEN={value}"),
            format!("password = \"{value}\""),
            format!("password: '{value}'"),
            format!("{{\"api_key\": \"{value}\"}}"),
            format!("\"client_secret\" : \"{value}\""),
            format!("db_password := \"{value}\""),
            format!("'token' => '{value}'"),
            format!("--password={value}"),
            format!("let token = \"{value}\";"),
            format!("auth_token: {value}"),
            format!("AWS_SECRET_ACCESS_KEY = {value}"),
            format!("secret=`{value}`"),
            format!("if token == \"{value}\" {{"),
            format!("if (token === '{value}') {{"),
        ];
        for text in &cases {
            assert_eq!(spans(text), [value.as_str()], "{text}");
            assert_eq!(only_finding(text).rule, "generic-secret");
        }
    }

    #[test]
    fn ordinary_code_and_prose_with_those_words_is_left_alone() {
        let hex = random_text(32, "0123456789abcdef", 110);
        let cases = [
            "let token = lexer.next_token_with_context_and_position();".to_string(),
            "token: TokenKind::Identifier,".to_string(),
            "pub secret_key: String,".to_string(),
            "password = input(\"Enter your password: \")".to_string(),
            "password = \"correct horse battery staple\"".to_string(),
            "max_tokens: 4096,".to_string(),
            "token_ids: Vec<u32>,".to_string(),
            "tokenizer_path = \"data/tokenizer.txt\"".to_string(),
            "api_key = os.environ[\"API_KEY\"]".to_string(),
            "api_key = None".to_string(),
            "secret = some_long_function_name_2".to_string(),
            "token = \"some_long_function_name_2\"".to_string(),
            "token_url = \"https://example.com/oauth/token/123456789abcdef\"".to_string(),
            "path_to_token = \"/home/user/.config/service/token123456\"".to_string(),
            "password = \"aaaaaaaaaaaaaaaaaaaa1\"".to_string(),
            "password = \"abababababababab12\"".to_string(),
            "password == \"x\"".to_string(),
            "token::new(abcdefghij0123456789)".to_string(),
            "secret = abcdefghij0123456789.method()".to_string(),
            "password = \"${DB_PASSWORD_VALUE_123}\"".to_string(),
            "password = \"{{ db_password_value_123 }}\"".to_string(),
            "The password must be at least 16 characters long, 12345678".to_string(),
            "passwords = [\"abcdefg\"]".to_string(),
            "token = \"12345678901234567890\"".to_string(),
            "token = \"abcdefghijklmnopqrstuvwxyz\"".to_string(),
        ];
        for text in &cases {
            assert!(rules(text).is_empty(), "{text:?} -> {:?}", scrubber().find(text));
        }
        // A hash is only caught next to a keyword, and then it is a secret.
        assert!(rules(&format!("sha256 = \"{hex}\"")).is_empty());
        assert_eq!(rules(&format!("secret = \"{hex}\"")), ["generic-secret"]);
    }

    #[test]
    fn unquoted_values_inside_argument_and_parameter_lists_are_code() {
        for text in [
            "        private_key: PKCS7PrivateKeyTypes,",
            "def sign(self, private_key: SSHCertPrivateKeyTypes) -> Certificate:",
            "connect(password=Secret1ValueXxXxXxXxXxXx, user=name)",
            "keys = [token1AbCdEfGhIjKlMnOp]",
        ] {
            assert!(rules(text).is_empty(), "{text:?} -> {:?}", scrubber().find(text));
        }
        let field = format!("let c = Config {{ secret: {} }};", "Abc1DefGhiJklMnoPqr");
        assert!(rules(&field).is_empty(), "{field:?} -> {:?}", scrubber().find(&field));
        // The same value at the end of a line, before a space, a comment, `;` or a quote is a
        // secret: that is how `.env` files, YAML and connection strings end.
        let value = random_text(36, ALNUM, 175);
        for text in [
            format!("token: {value}"),
            format!("token: {value} # rotate monthly"),
            format!("Server=db;Password={value};User=app"),
            format!("TOKEN={value}\nNEXT=1"),
            format!("sh -c 'export TOKEN={value}'"),
        ] {
            assert_eq!(spans(&text), [value.as_str()], "{text}");
        }
    }

    #[test]
    fn a_secret_that_starts_with_a_slash_is_still_a_secret_but_a_path_is_not() {
        // Standard base64 has `/` in its alphabet, so about one secret in 64 starts with it.
        let body = random_text(39, ALNUM, 170);
        let secret = format!("/{body}");
        assert_eq!(spans(&format!("aws_secret_access_key = {secret}")), [secret.as_str()]);
        assert_eq!(spans(&format!("secret = \"{secret}\"")), [secret.as_str()]);
        for path in [
            "token_dir = \"/home/alice123/projects/app\"",
            "token_dir = \"/usr/local/lib/x86_64-linux-gnu/lib1\"",
            "secret_path = \"/Users/Alice/.config/Service1/credentials\"",
            "secret_path = \"/Run/Secrets/Service_Account_Key_01\"",
            "token_file = ./config/Tokens1234567890.json",
            "token_file = \"~/.Config/Tokens1234567890\"",
        ] {
            assert!(rules(path).is_empty(), "{path:?} -> {:?}", scrubber().find(path));
        }
    }

    #[test]
    fn the_structured_rules_have_exact_length_limits() {
        // A JWT part needs 17 characters, `eyJ` included.
        let part = |n: usize, seed: u64| format!("eyJ{}", random_text(n - 3, URL_SAFE, seed));
        let signature = random_text(20, URL_SAFE, 181);
        assert_eq!(rules(&format!("x={}.{}.{signature}", part(17, 182), part(17, 183))), ["jwt"]);
        assert!(rules(&format!("x={}.{}.{signature}", part(16, 184), part(17, 185))).is_empty());
        assert!(rules(&format!("x={}.{}.{signature}", part(17, 186), part(16, 187))).is_empty());

        // An authorization credential needs 16 characters.
        let auth =
            |n: usize| format!("Authorization: Bearer {}", random_text(n, ALNUM, 188 + n as u64));
        assert_eq!(rules(&auth(16)), ["auth-header"]);
        assert!(rules(&auth(15)).is_empty());

        // A private key body needs a first run of 40 base64 characters and 64 in all.
        let alphabet = format!("{ALNUM}+/");
        let key = |first: usize, second: usize| {
            format!(
                "{DASHES}BEGIN PRIVATE KEY{DASHES}\n{}\n{}\n{DASHES}END PRIVATE KEY{DASHES}",
                random_text(first, &alphabet, 190),
                random_text(second, &alphabet, 191)
            )
        };
        assert_eq!(rules(&key(40, 24)), ["private-key"]);
        assert_eq!(rules(&key(41, 23)), ["private-key"]);
        assert!(rules(&key(40, 23)).is_empty(), "63 characters in all");
        assert!(rules(&key(39, 25)).is_empty(), "a first run of 39");
    }

    #[test]
    fn the_entropy_threshold_sits_between_3_45_and_3_58_bits() {
        // 16 characters: counts 3, 2, 2 and nine singles give 3.453 bits per character;
        // counts 3, 2 and eleven singles give 3.578. The rule cuts at 3.5.
        let below = "xxxyyzz123456789";
        let above = "xxxyy1234567890a";
        assert!(entropy(below.as_bytes()) < 3.5 && entropy(above.as_bytes()) > 3.5);
        assert!(rules(&format!("token = \"{}\"", below)).is_empty());
        assert_eq!(rules(&format!("token = \"{}\"", above)), ["generic-secret"]);
    }

    #[test]
    fn the_longer_of_two_matches_at_the_same_place_wins() {
        // The whole quoted value is replaced, not just the token at its start.
        let token = format!("{}{}", ["gh", "p_"].concat(), random_text(36, ALNUM, 200));
        let value = format!("{token}.suffix1234");
        let text = format!("secret = \"{value}\"");
        assert_eq!(spans(&text), [value.as_str()]);
        assert_eq!(only_finding(&text).rule, "generic-secret");
    }

    #[test]
    fn paths_and_namespaces_are_not_assignments() {
        let name = "Abc1Def2Ghi3Jkl4Mno5";
        for text in [
            format!("use crate::secret::{name};"),
            format!("let t = token::{name};"),
            format!("token_file = \"./config/Tokens1234567890.json\"{}", ""),
            format!("token_file = \"../keys/{name}.pem\""),
        ] {
            assert!(rules(&text).is_empty(), "{text:?} -> {:?}", scrubber().find(&text));
        }
    }

    #[test]
    fn only_a_real_scheme_separator_starts_a_url_password() {
        let pw = random_text(12, ALNUM, 201);
        // `:/x` is not `://`, even though the user and password after it have the right shape.
        let text = format!("a:/xuser:{pw}@host");
        assert!(rules(&text).is_empty(), "{text}");
        let text = format!("a://user:{pw}@host");
        assert_eq!(spans(&text), [pw.as_str()]);
    }

    #[test]
    fn a_credential_must_be_separated_from_its_scheme_by_a_blank() {
        let credential = random_text(30, ALNUM, 202);
        for text in [
            format!("Authorization: Bearer:{credential}"),
            format!("Authorization: Bearer/{credential}"),
            format!("Authorization: Bearer={credential}"),
        ] {
            assert!(rules(&text).is_empty(), "{text}");
        }
        assert_eq!(spans(&format!("Authorization: Bearer  {credential}")), [credential.as_str()]);
        assert_eq!(spans(&format!("Authorization: Bearer\t{credential}")), [credential.as_str()]);
    }

    #[test]
    fn a_key_body_is_searched_up_to_a_limit() {
        let header = format!("{DASHES}BEGIN PRIVATE KEY{DASHES}");
        let mut text = format!("{header}\n");
        let alphabet = format!("{ALNUM}+/");
        for n in 0..400 {
            text.push_str(&random_text(64, &alphabet, 300 + n));
            text.push('\n');
        }
        assert!(text.len() > header.len() + MAX_KEY_BODY + 1000);
        let found = only_finding(&text);
        assert_eq!((found.start, found.end), (0, header.len() + MAX_KEY_BODY));
    }

    #[test]
    fn a_label_that_is_prose_is_not_a_key_header() {
        let body = random_text(64, &format!("{ALNUM}+/"), 203);
        let text = format!(
            "{DASHES}BEGIN my notes on the PRIVATE KEY{DASHES}\n{body}\n{DASHES}END PRIVATE KEY{DASHES}"
        );
        assert!(rules(&text).is_empty(), "{text}");
        let text =
            format!("{DASHES}BEGIN PRIVATE KEY{DASHES}\n{body}\n{DASHES}END PRIVATE KEY{DASHES}");
        assert_eq!(rules(&text), ["private-key"]);
    }

    #[test]
    fn six_dashes_before_begin_still_find_the_key() {
        let body = random_text(64, &format!("{ALNUM}+/"), 204);
        let key =
            format!("{DASHES}BEGIN PRIVATE KEY{DASHES}\n{body}\n{DASHES}END PRIVATE KEY{DASHES}");
        let text = format!("-{key}");
        assert_eq!(spans(&text), [key.as_str()]);
    }

    #[test]
    fn the_generic_rule_has_length_limits() {
        let at = |n: usize| format!("token = \"{}\"", random_text(n, ALNUM, 120 + n as u64));
        assert!(rules(&at(15)).is_empty());
        assert_eq!(rules(&at(16)), ["generic-secret"]);
        assert_eq!(rules(&at(256)), ["generic-secret"]);
        assert!(rules(&at(257)).is_empty());
    }

    #[test]
    fn without_generic_keeps_the_rules_for_known_shapes() {
        let value = random_text(32, ALNUM, 130);
        let plain = format!("password = \"{value}\"");
        let scrubber = Scrubber::new().without_generic();
        assert!(scrubber.find(&plain).is_empty());
        assert_eq!(Scrubber::new().find(&plain).len(), 1);
        let token = format!("{}{}", ["gh", "p_"].concat(), random_text(36, ALNUM, 131));
        assert_eq!(scrubber.find(&format!("x {token}")).len(), 1);
        let header = format!("Authorization: Bearer {}", random_text(30, ALNUM, 132));
        assert_eq!(scrubber.find(&header).len(), 1);
    }

    #[test]
    fn where_rules_overlap_the_known_shape_wins() {
        let token = format!("{}{}", ["gh", "p_"].concat(), random_text(36, ALNUM, 140));
        let text = format!("GITHUB_TOKEN={token}");
        let found = scrubber().find(&text);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].rule, "github-token");
        // A bearer credential after `token =` is the auth rule's, not the generic one's.
        let value = random_text(30, ALNUM, 141);
        let text = format!("Authorization = \"Bearer {value}\"");
        assert_eq!(only_finding(&text).rule, "auth-header");
    }

    #[test]
    fn findings_are_sorted_and_do_not_overlap_and_offsets_refer_to_the_original() {
        let a = format!("{}{}", ["gh", "p_"].concat(), random_text(36, ALNUM, 150));
        let b = format!("{}{}", ["AK", "IA"].concat(), random_text(16, BASE32, 151));
        let text = format!("one {a} two\nthree {b} four\n");
        let found = scrubber().find(&text);
        assert_eq!(found.len(), 2);
        assert!(found[0].end <= found[1].start);
        assert_eq!(&text[found[0].start..found[0].end], a);
        assert_eq!(&text[found[1].start..found[1].end], b);
        let scrubbed = scrubber().scrub(&text);
        assert_eq!(scrubbed.text, format!("one {REDACTION} two\nthree {REDACTION} four\n"));
        assert_eq!(scrubbed.findings, found);
    }

    #[test]
    fn text_without_secrets_comes_back_unchanged() {
        let scrubbed = scrubber().scrub("fn main() {\n    println!(\"h\u{e9}llo\");\n}\n");
        assert_eq!(scrubbed.text, "fn main() {\n    println!(\"h\u{e9}llo\");\n}\n");
        assert!(scrubbed.findings.is_empty());
        assert_eq!(scrubber().scrub("").text, "");
    }

    #[test]
    fn scrubbing_is_idempotent() {
        let value = random_text(32, ALNUM, 160);
        let token = format!("{}{}", ["gh", "p_"].concat(), random_text(36, ALNUM, 161));
        let key = pem("RSA PRIVATE KEY", 6, 162, "\n");
        let text = format!(
            "A={value}\nGITHUB_TOKEN={token}\npassword = \"{value}\"\nhttps://u:{value}@h.example/x\n{key}\nAuthorization: Bearer {value}\n"
        );
        let once = scrubber().scrub(&text);
        assert_eq!(once.findings.len(), 5, "{:?}", once.findings);
        let twice = scrubber().scrub(&once.text);
        assert!(twice.findings.is_empty(), "{:?}", twice.findings);
        assert_eq!(twice.text, once.text);
    }

    #[test]
    fn the_crates_own_sources_hold_no_secrets() {
        // A false-positive check on real code: the rules must not fire on these files, which
        // mention every prefix and keyword the rules know.
        for (name, source) in [
            ("scrub.rs", include_str!("scrub.rs")),
            ("license.rs", include_str!("license.rs")),
            ("manifest.rs", include_str!("manifest.rs")),
            ("dedup.rs", include_str!("dedup.rs")),
            ("hash.rs", include_str!("hash.rs")),
            ("lib.rs", include_str!("lib.rs")),
        ] {
            let found = scrubber().find(source);
            assert!(
                found.is_empty(),
                "{name}: {:?}",
                found.iter().map(|f| (&source[f.start..f.end], f.rule)).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn random_text_with_secret_fragments_never_breaks_the_invariants() {
        let fragments = [
            "ghp_",
            "AKIA",
            "sk-",
            "sk-ant-",
            "eyJ",
            ".",
            "-----BEGIN PRIVATE KEY-----",
            "-----END PRIVATE KEY-----",
            "://",
            ":",
            "@",
            "password",
            "token",
            "=",
            "\"",
            "'",
            " ",
            "\n",
            "\\n",
            "Authorization",
            "Bearer ",
            "xoxb-",
            "hf_",
            "SG.",
            "\u{e9}",
            "\u{4e2d}",
            "\u{1f600}",
            "<",
            ">",
            "${",
            "}",
            "(",
            "/",
        ];
        let filler = ALNUM;
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let scrubber = scrubber();
        for round in 0..400 {
            let mut text = String::new();
            for _ in 0..(next() % 40) + 1 {
                if next() % 3 == 0 {
                    text.push_str(&random_text((next() % 70) as usize, filler, next()));
                } else {
                    text.push_str(fragments[(next() % fragments.len() as u64) as usize]);
                }
            }
            let found = scrubber.find(&text);
            let mut last_end = 0;
            for f in &found {
                assert!(f.start < f.end && f.end <= text.len(), "round {round}: {f:?}");
                assert!(f.start >= last_end, "round {round}: overlap {found:?}");
                assert!(
                    text.is_char_boundary(f.start) && text.is_char_boundary(f.end),
                    "round {round}: {f:?} in {text:?}"
                );
                last_end = f.end;
            }
            let scrubbed = scrubber.scrub(&text);
            assert_eq!(scrubbed.findings, found);
            // Everything outside the findings is kept, in order.
            let mut expected = String::new();
            let mut at = 0;
            for f in &found {
                expected.push_str(&text[at..f.start]);
                expected.push_str(REDACTION);
                at = f.end;
            }
            expected.push_str(&text[at..]);
            assert_eq!(scrubbed.text, expected);
            // Idempotent.
            let again = scrubber.scrub(&scrubbed.text);
            assert!(
                again.findings.is_empty(),
                "round {round}: {text:?} -> {:?} -> {:?}",
                scrubbed.text,
                again.findings
            );
        }
    }

    #[test]
    fn entropy_of_known_strings() {
        assert_eq!(entropy(b"aaaa"), 0.0);
        assert!((entropy(b"abab") - 1.0).abs() < 1e-12);
        assert!((entropy(b"abcdefgh") - 3.0).abs() < 1e-12);
    }
}
