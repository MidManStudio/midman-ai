// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-foundation.md, section "error.rs"
// ============================================================================
//! The error type shared by every MidMan crate.

use std::fmt;

/// Result alias used across the MidMan workspace.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors shared by the MidMan crates.
///
/// Variants carry text instead of wrapped source errors, so the type stays
/// `Clone` and `Eq` and test assertions stay simple.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A configuration value is invalid or missing.
    Config {
        /// The configuration field the problem is about.
        field: String,
        /// What is wrong with it.
        reason: String,
    },
    /// Tensor shapes, ranks or sizes do not fit an operation.
    Shape {
        /// The operation that rejected the shapes.
        op: String,
        /// What did not fit.
        detail: String,
    },
    /// An index (token id, row, class) is outside its valid range.
    IndexOutOfRange {
        /// What was being indexed.
        what: String,
        /// The offending index.
        index: usize,
        /// The exclusive upper bound.
        bound: usize,
    },
    /// Text input could not be parsed.
    Parse {
        /// What was being parsed.
        what: String,
        /// One-based line number, or 0 when no line applies.
        line: usize,
        /// What went wrong.
        message: String,
    },
    /// A format version is not supported by this build.
    UnsupportedVersion {
        /// The format whose version was checked.
        what: String,
        /// The version found in the input.
        found: u32,
        /// The version this build supports.
        supported: u32,
    },
    /// An operation was called in a way that cannot work, for example
    /// `backward` on a tensor that does not track gradients.
    InvalidArgument(String),
    /// An I/O failure, stored as text.
    Io(String),
}

impl Error {
    /// Builds an [`Error::Config`].
    pub fn config(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Error::Config { field: field.into(), reason: reason.into() }
    }

    /// Builds an [`Error::Shape`].
    pub fn shape(op: impl Into<String>, detail: impl Into<String>) -> Self {
        Error::Shape { op: op.into(), detail: detail.into() }
    }

    /// Builds an [`Error::IndexOutOfRange`].
    pub fn index(what: impl Into<String>, index: usize, bound: usize) -> Self {
        Error::IndexOutOfRange { what: what.into(), index, bound }
    }

    /// Builds an [`Error::Parse`]. Pass 0 as `line` when no line applies.
    pub fn parse(what: impl Into<String>, line: usize, message: impl Into<String>) -> Self {
        Error::Parse { what: what.into(), line, message: message.into() }
    }

    /// Builds an [`Error::InvalidArgument`].
    pub fn invalid(message: impl Into<String>) -> Self {
        Error::InvalidArgument(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config { field, reason } => {
                write!(f, "invalid config field `{field}`: {reason}")
            }
            Error::Shape { op, detail } => write!(f, "shape error in {op}: {detail}"),
            Error::IndexOutOfRange { what, index, bound } => {
                write!(f, "{what} index {index} is out of range (must be below {bound})")
            }
            Error::Parse { what, line, message } => {
                if *line > 0 {
                    write!(f, "failed to parse {what} at line {line}: {message}")
                } else {
                    write!(f, "failed to parse {what}: {message}")
                }
            }
            Error::UnsupportedVersion { what, found, supported } => {
                write!(
                    f,
                    "{what} version {found} is not supported (this build supports {supported})"
                )
            }
            Error::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Error::Io(message) => write!(f, "I/O error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_specific() {
        assert_eq!(
            Error::config("d_model", "must be positive").to_string(),
            "invalid config field `d_model`: must be positive"
        );
        assert_eq!(
            Error::shape("matmul", "inner dimensions 3 and 4 differ").to_string(),
            "shape error in matmul: inner dimensions 3 and 4 differ"
        );
        assert_eq!(
            Error::index("token id", 9, 4).to_string(),
            "token id index 9 is out of range (must be below 4)"
        );
        assert_eq!(
            Error::parse("model config", 3, "expected `key = value`").to_string(),
            "failed to parse model config at line 3: expected `key = value`"
        );
        assert_eq!(
            Error::parse("model config", 0, "empty input").to_string(),
            "failed to parse model config: empty input"
        );
        assert_eq!(
            Error::UnsupportedVersion { what: "checkpoint".into(), found: 2, supported: 1 }
                .to_string(),
            "checkpoint version 2 is not supported (this build supports 1)"
        );
        assert_eq!(Error::invalid("not a scalar").to_string(), "invalid argument: not a scalar");
    }

    #[test]
    fn io_errors_convert_to_text() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let err: Error = io.into();
        assert_eq!(err, Error::Io("no such file".to_string()));
    }

    #[test]
    fn errors_are_cloneable_and_comparable() {
        let a = Error::shape("add", "rank mismatch");
        assert_eq!(a.clone(), a);
        assert_ne!(a, Error::shape("add", "other"));
    }
}
