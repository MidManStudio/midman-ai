// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "license.rs"
// ============================================================================
//! SPDX license expressions and the license policy for training data.

use std::collections::BTreeSet;
use std::fmt;

use midman_foundation::{Error, Result};

const WHAT: &str = "license expression";

/// Longest expression accepted, in bytes. Real ones are under 100.
const MAX_EXPRESSION_BYTES: usize = 1024;

/// Deepest nesting of parentheses accepted.
const MAX_DEPTH: usize = 16;

/// Licenses counted as permissive by the default policy (SPDX identifiers).
///
/// The choice of this list is the project's own; the user's decision was "permissive plus
/// weak copyleft". Every entry is a current, non-deprecated identifier of the SPDX list.
pub const PERMISSIVE: &[&str] = &[
    "0BSD",
    "Apache-2.0",
    "BlueOak-1.0.0",
    "BSD-1-Clause",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSD-3-Clause-Clear",
    "BSL-1.0",
    "CC0-1.0",
    "curl",
    "HPND",
    "ISC",
    "Libpng",
    "MIT",
    "MIT-0",
    "NCSA",
    "PSF-2.0",
    "Python-2.0",
    "Unicode-3.0",
    "Unicode-DFS-2016",
    "Unlicense",
    "UPL-1.0",
    "WTFPL",
    "X11",
    "Zlib",
];

/// The weak copyleft licenses of the default policy: the MPL and LGPL families.
pub const WEAK_COPYLEFT: &[&str] = &[
    "LGPL-2.0-only",
    "LGPL-2.0-or-later",
    "LGPL-2.1-only",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "MPL-1.0",
    "MPL-1.1",
    "MPL-2.0",
    "MPL-2.0-no-copyleft-exception",
];

/// A parsed SPDX license expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseExpr {
    /// One license: an identifier, an optional `+` ("or any later version") and an optional
    /// `WITH` exception.
    License {
        /// The identifier as written, for example `Apache-2.0` or `LicenseRef-in-house`.
        id: String,
        /// Whether the identifier was followed by `+`.
        or_later: bool,
        /// The exception after `WITH`, as written.
        exception: Option<String>,
    },
    /// All of the parts apply.
    And(Vec<LicenseExpr>),
    /// The user may choose any one of the parts.
    Or(Vec<LicenseExpr>),
}

impl LicenseExpr {
    /// Parses an SPDX license expression.
    ///
    /// The grammar is the SPDX 2.3 one: identifiers, `ID+`, `ID WITH EXCEPTION`, `AND`, `OR`
    /// and parentheses, with `WITH` binding tighter than `AND` and `AND` tighter than `OR`.
    /// The operators must be written in capitals, as the specification requires. The
    /// identifiers are not looked up in the SPDX list.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Parse`] for text that is not an expression, for lowercase
    /// operators, for an expression longer than 1,024 bytes and for nesting deeper than 16.
    pub fn parse(text: &str) -> Result<LicenseExpr> {
        if text.len() > MAX_EXPRESSION_BYTES {
            return Err(fail(format!("longer than {MAX_EXPRESSION_BYTES} bytes")));
        }
        let tokens = lex(text)?;
        if tokens.is_empty() {
            return Err(fail("the expression is empty".to_string()));
        }
        let mut parser = Parser { tokens, pos: 0 };
        let expr = parser.parse_or(0)?;
        match parser.peek() {
            None => Ok(expr),
            Some(token) => {
                Err(fail(format!("unexpected {} after the expression", token.describe())))
            }
        }
    }
}

impl fmt::Display for LicenseExpr {
    /// Writes the expression with the parentheses needed to read it back the same way.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LicenseExpr::License { id, or_later, exception } => {
                write!(f, "{id}")?;
                if *or_later {
                    write!(f, "+")?;
                }
                if let Some(exception) = exception {
                    write!(f, " WITH {exception}")?;
                }
                Ok(())
            }
            LicenseExpr::And(parts) => write_joined(f, parts, " AND ", |p| {
                matches!(p, LicenseExpr::And(_) | LicenseExpr::Or(_))
            }),
            LicenseExpr::Or(parts) => {
                write_joined(f, parts, " OR ", |p| matches!(p, LicenseExpr::Or(_)))
            }
        }
    }
}

fn write_joined(
    f: &mut fmt::Formatter<'_>,
    parts: &[LicenseExpr],
    separator: &str,
    needs_parentheses: impl Fn(&LicenseExpr) -> bool,
) -> fmt::Result {
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            write!(f, "{separator}")?;
        }
        if needs_parentheses(part) {
            write!(f, "({part})")?;
        } else {
            write!(f, "{part}")?;
        }
    }
    Ok(())
}

fn fail(message: String) -> Error {
    Error::parse(WHAT, 0, message)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Open,
    Close,
    And,
    Or,
    With,
    Id { text: String, plus: bool },
}

impl Token {
    fn describe(&self) -> String {
        match self {
            Token::Open => "`(`".to_string(),
            Token::Close => "`)`".to_string(),
            Token::And => "`AND`".to_string(),
            Token::Or => "`OR`".to_string(),
            Token::With => "`WITH`".to_string(),
            Token::Id { text, .. } => format!("`{text}`"),
        }
    }
}

fn lex(text: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    for ch in text.chars() {
        if ch == '(' || ch == ')' || ch.is_ascii_whitespace() {
            flush_word(&mut word, &mut tokens)?;
            match ch {
                '(' => tokens.push(Token::Open),
                ')' => tokens.push(Token::Close),
                _ => {}
            }
        } else {
            word.push(ch);
        }
    }
    flush_word(&mut word, &mut tokens)?;
    Ok(tokens)
}

fn flush_word(word: &mut String, tokens: &mut Vec<Token>) -> Result<()> {
    if word.is_empty() {
        return Ok(());
    }
    let text = std::mem::take(word);
    let token = match text.as_str() {
        "AND" => Token::And,
        "OR" => Token::Or,
        "WITH" => Token::With,
        _ if text.eq_ignore_ascii_case("and")
            || text.eq_ignore_ascii_case("or")
            || text.eq_ignore_ascii_case("with") =>
        {
            return Err(fail(format!("`{text}` must be written in capitals (AND, OR, WITH)")));
        }
        _ => {
            let (body, plus) = match text.strip_suffix('+') {
                Some(body) => (body.to_string(), true),
                None => (text.clone(), false),
            };
            check_identifier(&body)?;
            if plus && is_reference(&body) {
                return Err(fail(format!("`{text}`: a reference cannot be followed by `+`")));
            }
            Token::Id { text: body, plus }
        }
    };
    tokens.push(token);
    Ok(())
}

/// Whether a word is a `LicenseRef-` or `AdditionRef-` reference, with or without the
/// `DocumentRef-` part.
fn is_reference(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    lower.starts_with("licenseref-")
        || lower.starts_with("additionref-")
        || lower.starts_with("documentref-")
}

fn is_id_string(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
}

/// The part of a `LicenseRef-` or `AdditionRef-` reference after its prefix, given in lowercase.
fn reference_body(lower: &str) -> Option<&str> {
    lower.strip_prefix("licenseref-").or_else(|| lower.strip_prefix("additionref-"))
}

/// Checks the characters of an identifier, exception or reference.
fn check_identifier(word: &str) -> Result<()> {
    let lower = word.to_ascii_lowercase();
    let ok = match lower.strip_prefix("documentref-") {
        Some(rest) => match rest.split_once(':') {
            Some((document, reference)) => {
                is_id_string(document) && reference_body(reference).is_some_and(is_id_string)
            }
            None => false,
        },
        None => match reference_body(&lower) {
            Some(body) => is_id_string(body),
            None => is_id_string(&lower),
        },
    };
    if ok {
        Ok(())
    } else {
        Err(fail(format!("`{word}` is not a valid license identifier")))
    }
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn parse_or(&mut self, depth: usize) -> Result<LicenseExpr> {
        let mut parts = vec![self.parse_and(depth)?];
        while self.peek() == Some(&Token::Or) {
            self.pos += 1;
            parts.push(self.parse_and(depth)?);
        }
        Ok(if parts.len() == 1 { parts.remove(0) } else { LicenseExpr::Or(parts) })
    }

    fn parse_and(&mut self, depth: usize) -> Result<LicenseExpr> {
        let mut parts = vec![self.parse_primary(depth)?];
        while self.peek() == Some(&Token::And) {
            self.pos += 1;
            parts.push(self.parse_primary(depth)?);
        }
        Ok(if parts.len() == 1 { parts.remove(0) } else { LicenseExpr::And(parts) })
    }

    fn parse_primary(&mut self, depth: usize) -> Result<LicenseExpr> {
        let token = match self.tokens.get(self.pos) {
            Some(token) => token.clone(),
            None => return Err(fail("the expression ends where a license was expected".into())),
        };
        self.pos += 1;
        match token {
            Token::Open => {
                if depth >= MAX_DEPTH {
                    return Err(fail(format!("parentheses are nested deeper than {MAX_DEPTH}")));
                }
                let inner = self.parse_or(depth + 1)?;
                match self.tokens.get(self.pos) {
                    Some(Token::Close) => {
                        self.pos += 1;
                        Ok(inner)
                    }
                    Some(other) => {
                        Err(fail(format!("expected `)` but found {}", other.describe())))
                    }
                    None => Err(fail("a `(` is never closed".into())),
                }
            }
            Token::Id { text, plus } => {
                let exception = if self.peek() == Some(&Token::With) {
                    self.pos += 1;
                    match self.tokens.get(self.pos) {
                        Some(Token::Id { text, plus: false }) => {
                            let exception = text.clone();
                            self.pos += 1;
                            Some(exception)
                        }
                        Some(other) => {
                            return Err(fail(format!(
                                "expected an exception after `WITH` but found {}",
                                other.describe()
                            )))
                        }
                        None => return Err(fail("`WITH` is not followed by an exception".into())),
                    }
                } else {
                    None
                };
                Ok(LicenseExpr::License { id: text, or_later: plus, exception })
            }
            other => Err(fail(format!("expected a license but found {}", other.describe()))),
        }
    }
}

/// The outcome of checking a license expression against a [`LicensePolicy`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The source may be used.
    Accepted,
    /// The source may not be used; the text says why.
    Rejected(String),
}

impl Verdict {
    /// Whether the verdict is [`Verdict::Accepted`].
    pub fn is_accepted(&self) -> bool {
        matches!(self, Verdict::Accepted)
    }
}

/// The set of licenses that training data may carry.
///
/// ```
/// use midman_data::{LicensePolicy, Verdict};
///
/// let policy = LicensePolicy::permissive_and_weak_copyleft();
/// assert!(policy.check("MIT OR Apache-2.0").unwrap().is_accepted());
/// assert!(policy.check("MPL-2.0").unwrap().is_accepted());
/// assert!(matches!(policy.check("GPL-3.0-only").unwrap(), Verdict::Rejected(_)));
/// ```
///
/// An `OR` is accepted if any alternative is, an `AND` only if every part is. An exception
/// (`WITH`) never turns a license that is not on the list into one that is: the license
/// part is judged on its own. Identifiers match without regard to case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LicensePolicy {
    allowed: BTreeSet<String>,
    unlicensed: bool,
}

impl LicensePolicy {
    /// A policy that allows nothing. Add licenses with [`LicensePolicy::allow`].
    pub fn empty() -> Self {
        LicensePolicy { allowed: BTreeSet::new(), unlicensed: false }
    }

    /// The default policy: [`PERMISSIVE`] plus [`WEAK_COPYLEFT`].
    pub fn permissive_and_weak_copyleft() -> Self {
        let mut policy = Self::empty();
        for id in PERMISSIVE.iter().chain(WEAK_COPYLEFT) {
            policy = policy.allow(id);
        }
        policy
    }

    /// Allows one more license, for example `LicenseRef-in-house` for the project's own
    /// data. A trailing `+` allows "this version or later", as in an expression.
    #[must_use]
    pub fn allow(mut self, id: &str) -> Self {
        let (body, plus) = match id.strip_suffix('+') {
            Some(body) => (body, true),
            None => (id, false),
        };
        self.allowed.insert(canonical_id(body, plus));
        self
    }

    /// Also accepts a source whose listing says it has no license: the SPDX value `NONE`.
    ///
    /// ```
    /// use midman_data::LicensePolicy;
    ///
    /// let strict = LicensePolicy::permissive_and_weak_copyleft();
    /// let open = strict.clone().allow_unlicensed();
    /// assert!(!strict.check("NONE").unwrap().is_accepted());
    /// assert!(open.check("NONE").unwrap().is_accepted());
    /// // A license that exists but could not be identified is not "no license".
    /// assert!(!open.check("NOASSERTION").unwrap().is_accepted());
    /// ```
    ///
    /// This is a choice about the corpus, not a statement of law: code with no license is
    /// not free to use by default, so the project decides, and a source that says anything
    /// in its own files against use has to be caught by a separate check (`ingest.py notices`).
    #[must_use]
    pub fn allow_unlicensed(mut self) -> Self {
        self.unlicensed = true;
        self
    }

    /// Whether `NONE` is accepted, see [`LicensePolicy::allow_unlicensed`].
    pub fn allows_unlicensed(&self) -> bool {
        self.unlicensed
    }

    /// The allowed identifiers in canonical lowercase form, sorted.
    pub fn allowed_ids(&self) -> impl Iterator<Item = &str> {
        self.allowed.iter().map(String::as_str)
    }

    /// Parses `expression` and checks it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Parse`] if `expression` is not a valid SPDX expression. A valid
    /// expression for a license that is not allowed is [`Verdict::Rejected`], not an error.
    pub fn check(&self, expression: &str) -> Result<Verdict> {
        Ok(self.check_expr(&LicenseExpr::parse(expression)?))
    }

    /// Checks an already parsed expression.
    pub fn check_expr(&self, expr: &LicenseExpr) -> Verdict {
        match expr {
            LicenseExpr::License { id, or_later, exception } => {
                let lower = id.to_ascii_lowercase();
                if exception.is_none() {
                    if lower == "none" {
                        return if self.unlicensed {
                            Verdict::Accepted
                        } else {
                            Verdict::Rejected(format!("no license was identified (`{id}`)"))
                        };
                    }
                    if lower == "noassertion" {
                        return Verdict::Rejected(format!(
                            "the license could not be identified (`{id}`)"
                        ));
                    }
                }
                if self.allowed.contains(&canonical_id(id, *or_later)) {
                    Verdict::Accepted
                } else {
                    Verdict::Rejected(format!("`{expr}` is not on the allow list"))
                }
            }
            LicenseExpr::And(parts) => {
                let reasons: Vec<String> = parts
                    .iter()
                    .filter_map(|part| match self.check_expr(part) {
                        Verdict::Accepted => None,
                        Verdict::Rejected(reason) => Some(reason),
                    })
                    .collect();
                if reasons.is_empty() {
                    Verdict::Accepted
                } else {
                    Verdict::Rejected(join_reasons(reasons))
                }
            }
            LicenseExpr::Or(parts) => {
                let mut reasons = Vec::new();
                for part in parts {
                    match self.check_expr(part) {
                        Verdict::Accepted => return Verdict::Accepted,
                        Verdict::Rejected(reason) => reasons.push(reason),
                    }
                }
                Verdict::Rejected(join_reasons(reasons))
            }
        }
    }
}

fn join_reasons(mut reasons: Vec<String>) -> String {
    reasons.dedup();
    reasons.join("; ")
}

/// The form in which identifiers are compared: lowercase, with the old GNU identifiers
/// spelled the current way.
///
/// `LGPL-2.1` alone is the deprecated spelling of `LGPL-2.1-only`, and `LGPL-2.1+` of
/// `LGPL-2.1-or-later`; the same holds for the GPL, AGPL and GFDL. For any other identifier a
/// trailing `+` is ignored.
fn canonical_id(id: &str, or_later: bool) -> String {
    let lower = id.to_ascii_lowercase();
    if is_bare_gnu_version(&lower) {
        let suffix = if or_later { "or-later" } else { "only" };
        return format!("{lower}-{suffix}");
    }
    lower
}

/// `gpl-2.0`, `lgpl-2.1`, `agpl-3.0`, `gfdl-1.3`: a GNU family name and a version, nothing
/// else.
fn is_bare_gnu_version(lower: &str) -> bool {
    ["gpl-", "lgpl-", "agpl-", "gfdl-"].iter().any(|family| {
        lower.strip_prefix(family).is_some_and(|version| {
            let mut parts = version.split('.');
            let major = parts.next().unwrap_or("");
            let minor = parts.next();
            parts.next().is_none()
                && !major.is_empty()
                && major.chars().all(|c| c.is_ascii_digit())
                && minor.is_some_and(|m| !m.is_empty() && m.chars().all(|c| c.is_ascii_digit()))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn license(id: &str) -> LicenseExpr {
        LicenseExpr::License { id: id.to_string(), or_later: false, exception: None }
    }

    fn policy() -> LicensePolicy {
        LicensePolicy::permissive_and_weak_copyleft()
    }

    fn accepted(expression: &str) -> bool {
        policy().check(expression).unwrap().is_accepted()
    }

    #[test]
    fn single_identifiers_parse() {
        assert_eq!(LicenseExpr::parse("MIT").unwrap(), license("MIT"));
        assert_eq!(LicenseExpr::parse("  Apache-2.0 \n").unwrap(), license("Apache-2.0"));
        assert_eq!(
            LicenseExpr::parse("LGPL-2.1+").unwrap(),
            LicenseExpr::License { id: "LGPL-2.1".into(), or_later: true, exception: None }
        );
        assert_eq!(
            LicenseExpr::parse("Apache-2.0 WITH LLVM-exception").unwrap(),
            LicenseExpr::License {
                id: "Apache-2.0".into(),
                or_later: false,
                exception: Some("LLVM-exception".into())
            }
        );
    }

    #[test]
    fn references_parse() {
        assert_eq!(
            LicenseExpr::parse("LicenseRef-in-house").unwrap(),
            license("LicenseRef-in-house")
        );
        assert_eq!(
            LicenseExpr::parse("DocumentRef-spdx-tool-1.2:LicenseRef-MIT-Style-2").unwrap(),
            license("DocumentRef-spdx-tool-1.2:LicenseRef-MIT-Style-2")
        );
    }

    #[test]
    fn precedence_is_with_then_and_then_or() {
        let expr = LicenseExpr::parse("MIT OR Apache-2.0 AND BSD-3-Clause").unwrap();
        assert_eq!(
            expr,
            LicenseExpr::Or(vec![
                license("MIT"),
                LicenseExpr::And(vec![license("Apache-2.0"), license("BSD-3-Clause")])
            ])
        );
        let expr = LicenseExpr::parse("A AND B WITH X").unwrap();
        assert_eq!(
            expr,
            LicenseExpr::And(vec![
                license("A"),
                LicenseExpr::License {
                    id: "B".into(),
                    or_later: false,
                    exception: Some("X".into())
                }
            ])
        );
    }

    #[test]
    fn parentheses_group() {
        let expr = LicenseExpr::parse("(MIT OR Apache-2.0) AND ISC").unwrap();
        assert_eq!(
            expr,
            LicenseExpr::And(vec![
                LicenseExpr::Or(vec![license("MIT"), license("Apache-2.0")]),
                license("ISC")
            ])
        );
        assert_eq!(LicenseExpr::parse("((MIT))").unwrap(), license("MIT"));
    }

    #[test]
    fn display_reads_back_as_the_same_expression() {
        for text in [
            "MIT",
            "MIT OR Apache-2.0",
            "MIT AND Apache-2.0",
            "MIT OR Apache-2.0 AND ISC",
            "(MIT OR Apache-2.0) AND ISC",
            "MIT OR (Apache-2.0 OR ISC)",
            "MIT AND (Apache-2.0 AND ISC)",
            "((A OR B) AND (C OR D)) OR E",
            "GPL-2.0+ WITH Classpath-exception-2.0 OR LGPL-3.0",
            "LicenseRef-x AND DocumentRef-d:LicenseRef-y",
        ] {
            let expr = LicenseExpr::parse(text).unwrap();
            let again = LicenseExpr::parse(&expr.to_string()).unwrap();
            assert_eq!(expr, again, "{text} -> {expr}");
        }
        assert_eq!(
            LicenseExpr::parse("(MIT OR Apache-2.0) AND ISC").unwrap().to_string(),
            "(MIT OR Apache-2.0) AND ISC"
        );
        assert_eq!(
            LicenseExpr::parse("MIT OR Apache-2.0 AND ISC").unwrap().to_string(),
            "MIT OR Apache-2.0 AND ISC"
        );
    }

    #[test]
    fn malformed_expressions_are_parse_errors() {
        for text in [
            "",
            "   ",
            "(",
            ")",
            "()",
            "MIT OR",
            "OR MIT",
            "MIT AND AND ISC",
            "MIT WITH",
            "MIT WITH WITH",
            "MIT WITH (X)",
            "(MIT OR ISC) WITH X",
            "MIT ISC",
            "MIT)",
            "(MIT",
            "MIT OR (ISC",
            "MIT +",
            "MIT++",
            "MIT/Apache-2.0",
            "MIT, Apache-2.0",
            "MIT_X",
            "LicenseRef-",
            "LicenseRef-x+",
            "DocumentRef-d",
            "DocumentRef-d:MIT",
            "DocumentRef-:LicenseRef-y",
            "MIT WITH X+",
            "caf\u{e9}",
        ] {
            let result = LicenseExpr::parse(text);
            assert!(matches!(result, Err(Error::Parse { .. })), "{text:?} gave {result:?}");
        }
    }

    #[test]
    fn lowercase_operators_are_refused_by_name() {
        for text in ["MIT or Apache-2.0", "MIT and ISC", "MIT with LLVM-exception", "MIT Or ISC"] {
            let message = LicenseExpr::parse(text).unwrap_err().to_string();
            assert!(message.contains("capitals"), "{text}: {message}");
        }
    }

    #[test]
    fn limits_on_size_and_depth() {
        let long = format!("MIT{}", " OR MIT".repeat(200));
        assert!(long.len() > MAX_EXPRESSION_BYTES);
        assert!(LicenseExpr::parse(&long).unwrap_err().to_string().contains("1024"));
        // The limit is exact: 1,024 bytes pass (padding is whitespace), 1,025 do not.
        let padded = |n: usize| format!("MIT{}", " ".repeat(n - 3));
        assert!(LicenseExpr::parse(&padded(MAX_EXPRESSION_BYTES)).is_ok());
        assert!(LicenseExpr::parse(&padded(MAX_EXPRESSION_BYTES + 1)).is_err());
        let at_depth = |n: usize| format!("{}MIT{}", "(".repeat(n), ")".repeat(n));
        assert!(LicenseExpr::parse(&at_depth(MAX_DEPTH)).is_ok());
        assert!(LicenseExpr::parse(&at_depth(MAX_DEPTH + 1)).is_err());
        // Hostile input of any depth is an error, never a stack overflow.
        assert!(LicenseExpr::parse(&"(".repeat(1000)).is_err());
    }

    #[test]
    fn error_messages_say_what_is_wrong() {
        let message = LicenseExpr::parse("MIT OR").unwrap_err().to_string();
        assert!(message.contains("license expression") && message.contains("ends"), "{message}");
        let message = LicenseExpr::parse("MIT/ISC").unwrap_err().to_string();
        assert!(message.contains("`MIT/ISC`"), "{message}");
    }

    #[test]
    fn default_list_is_accepted_and_nothing_else_by_accident() {
        let policy = policy();
        for id in PERMISSIVE.iter().chain(WEAK_COPYLEFT) {
            assert!(policy.check(id).unwrap().is_accepted(), "{id}");
        }
        assert_eq!(policy.allowed_ids().count(), PERMISSIVE.len() + WEAK_COPYLEFT.len());
    }

    #[test]
    fn strong_copyleft_and_unknown_licenses_are_rejected() {
        for id in [
            "GPL-2.0-only",
            "GPL-3.0-or-later",
            "GPL-3.0",
            "GPL-2.0+",
            "AGPL-3.0-only",
            "AGPL-3.0-or-later",
            "GFDL-1.3-only",
            "SSPL-1.0",
            "BUSL-1.1",
            "CC-BY-NC-4.0",
            "EPL-2.0",
            "CDDL-1.0",
            "EUPL-1.2",
            "Proprietary",
            "other",
            "LicenseRef-whatever",
        ] {
            assert!(!accepted(id), "{id} should be rejected");
        }
    }

    #[test]
    fn identifiers_match_without_regard_to_case() {
        assert!(accepted("mit"));
        assert!(accepted("APACHE-2.0"));
        assert!(accepted("lgpl-2.1-OR-LATER"));
        assert!(!accepted("gpl-3.0-only"));
    }

    #[test]
    fn old_gnu_spellings_mean_only_and_or_later() {
        // LGPL-2.1 alone is LGPL-2.1-only; with + it is LGPL-2.1-or-later. Both are allowed.
        assert!(accepted("LGPL-2.1"));
        assert!(accepted("LGPL-2.1+"));
        assert!(accepted("LGPL-3.0"));
        assert!(accepted("LGPL-2.0+"));
        // A policy that lists only the "only" form does not accept the "or later" one.
        let only = LicensePolicy::empty().allow("LGPL-2.1-only");
        assert!(only.check("LGPL-2.1").unwrap().is_accepted());
        assert!(!only.check("LGPL-2.1+").unwrap().is_accepted());
        assert!(!only.check("LGPL-2.1-or-later").unwrap().is_accepted());
        let later = LicensePolicy::empty().allow("LGPL-2.1+");
        assert!(later.check("LGPL-2.1-or-later").unwrap().is_accepted());
        assert!(!later.check("LGPL-2.1").unwrap().is_accepted());
    }

    #[test]
    fn plus_on_other_identifiers_is_ignored() {
        assert!(accepted("Apache-2.0+"));
        assert!(!accepted("SSPL-1.0+"));
    }

    #[test]
    fn or_needs_one_alternative_and_needs_all_parts() {
        assert!(accepted("MIT OR GPL-3.0-only"));
        assert!(accepted("GPL-3.0-only OR MIT"));
        assert!(!accepted("GPL-3.0-only OR AGPL-3.0-only"));
        assert!(accepted("MIT AND Apache-2.0"));
        assert!(!accepted("MIT AND GPL-3.0-only"));
        assert!(accepted("(MIT AND GPL-3.0-only) OR ISC"));
        assert!(!accepted("(MIT OR ISC) AND GPL-3.0-only"));
        assert!(accepted("MPL-2.0 AND LGPL-2.1-or-later AND MIT"));
    }

    #[test]
    fn an_exception_never_rescues_a_license_that_is_not_allowed() {
        assert!(accepted("Apache-2.0 WITH LLVM-exception"));
        assert!(accepted("MIT WITH Anything-exception"));
        assert!(!accepted("GPL-2.0-only WITH Classpath-exception-2.0"));
        assert!(!accepted("GPL-3.0-or-later WITH GCC-exception-3.1"));
        assert!(!accepted("GPL-2.0+ WITH Classpath-exception-2.0"));
    }

    #[test]
    fn no_license_found_is_rejected_with_its_own_reason() {
        for id in ["NONE", "none"] {
            match policy().check(id).unwrap() {
                Verdict::Rejected(reason) => {
                    assert!(reason.starts_with("no license was identified"), "{reason}")
                }
                Verdict::Accepted => panic!("{id} accepted"),
            }
        }
        for id in ["NOASSERTION", "noassertion"] {
            match policy().check(id).unwrap() {
                Verdict::Rejected(reason) => {
                    assert!(reason.starts_with("the license could not be identified"), "{reason}")
                }
                Verdict::Accepted => panic!("{id} accepted"),
            }
        }
    }

    #[test]
    fn unlicensed_is_accepted_only_when_the_policy_says_so() {
        assert!(!policy().allows_unlicensed());
        let open = policy().allow_unlicensed();
        assert!(open.allows_unlicensed());
        for id in ["NONE", "none", "None"] {
            assert!(open.check(id).unwrap().is_accepted(), "{id}");
        }
        // Unidentified is a different thing from none, and stays refused.
        for id in ["NOASSERTION", "noassertion"] {
            assert!(!open.check(id).unwrap().is_accepted(), "{id}");
        }
        // The rest of the policy is unchanged.
        assert!(open.check("MIT").unwrap().is_accepted());
        assert!(!open.check("GPL-3.0-only").unwrap().is_accepted());
        assert!(!open.check("GPL-3.0-only AND NONE").unwrap().is_accepted());
        assert!(!open.check("NONE AND NOASSERTION").unwrap().is_accepted());
        assert!(open.check("NONE OR GPL-3.0-only").unwrap().is_accepted());
        // An exception does not turn NONE into a license; it is judged as a name.
        assert!(!open.check("NONE WITH LLVM-exception").unwrap().is_accepted());
        // It is a flag of the policy, not an allowed identifier.
        assert!(!LicensePolicy::empty().allow_unlicensed().check("MIT").unwrap().is_accepted());
        assert!(LicensePolicy::empty().allow_unlicensed().check("NONE").unwrap().is_accepted());
        assert_eq!(open.allowed_ids().count(), policy().allowed_ids().count());
        assert_ne!(open, policy());
    }

    #[test]
    fn rejection_reasons_name_the_licenses() {
        match policy().check("GPL-3.0-only").unwrap() {
            Verdict::Rejected(reason) => {
                assert_eq!(reason, "`GPL-3.0-only` is not on the allow list")
            }
            Verdict::Accepted => panic!(),
        }
        match policy().check("GPL-2.0-only WITH Classpath-exception-2.0").unwrap() {
            Verdict::Rejected(reason) => {
                assert!(reason.contains("GPL-2.0-only WITH Classpath-exception-2.0"), "{reason}")
            }
            Verdict::Accepted => panic!(),
        }
        match policy().check("GPL-2.0-only AND SSPL-1.0").unwrap() {
            Verdict::Rejected(reason) => {
                assert!(reason.contains("GPL-2.0-only") && reason.contains("SSPL-1.0"), "{reason}")
            }
            Verdict::Accepted => panic!(),
        }
        match policy().check("GPL-2.0-only OR SSPL-1.0").unwrap() {
            Verdict::Rejected(reason) => {
                assert!(reason.contains("GPL-2.0-only") && reason.contains("SSPL-1.0"), "{reason}")
            }
            Verdict::Accepted => panic!(),
        }
    }

    #[test]
    fn a_reference_can_be_allowed_for_the_projects_own_data() {
        let policy = policy().allow("LicenseRef-in-house");
        assert!(policy.check("LicenseRef-in-house").unwrap().is_accepted());
        assert!(policy.check("licenseref-IN-HOUSE").unwrap().is_accepted());
        assert!(!policy.check("LicenseRef-other").unwrap().is_accepted());
        assert!(!LicensePolicy::permissive_and_weak_copyleft()
            .check("LicenseRef-in-house")
            .unwrap()
            .is_accepted());
    }

    #[test]
    fn the_empty_policy_allows_nothing() {
        let empty = LicensePolicy::empty();
        assert!(!empty.check("MIT").unwrap().is_accepted());
        assert_eq!(empty.allowed_ids().count(), 0);
    }

    #[test]
    fn syntax_errors_are_errors_not_rejections() {
        assert!(policy().check("MIT OR").is_err());
        assert!(policy().check("").is_err());
    }

    #[test]
    fn allowed_ids_are_canonical_lowercase_and_sorted() {
        let policy = LicensePolicy::empty().allow("MIT").allow("LGPL-2.1").allow("Apache-2.0");
        let ids: Vec<&str> = policy.allowed_ids().collect();
        assert_eq!(ids, ["apache-2.0", "lgpl-2.1-only", "mit"]);
    }
}
