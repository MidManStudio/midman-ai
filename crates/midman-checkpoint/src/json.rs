// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-checkpoint.md, section "json.rs"
// ============================================================================
//! A small JSON reader and string writer: just enough for a safetensors header.
//!
//! A header holds only objects, arrays, strings and non-negative integers, so that is
//! all the reader accepts. Booleans, `null`, negative numbers, fractions and exponents
//! are errors, which keeps a corrupt or hostile header from being half understood.

use std::collections::HashSet;

use midman_foundation::{Error, Result};

/// What the errors say was being parsed.
const WHAT: &str = "safetensors header";

/// Objects and arrays may nest this deep. A header needs three levels.
const MAX_DEPTH: usize = 8;

/// A parsed JSON value, restricted to what a safetensors header can contain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Value {
    /// Key and value pairs in the order they appeared. Keys are unique.
    Object(Vec<(String, Value)>),
    /// Values in order.
    Array(Vec<Value>),
    /// Unescaped text.
    String(String),
    /// A non-negative integer.
    Number(u64),
}

/// Parses `text` as one JSON value, allowing whitespace before and after it.
pub(crate) fn parse(text: &str) -> Result<Value> {
    let mut parser = Parser { bytes: text.as_bytes(), pos: 0 };
    parser.skip_whitespace();
    let value = parser.value(0)?;
    parser.skip_whitespace();
    if parser.pos != parser.bytes.len() {
        return Err(parser.fail("unexpected data after the top-level value"));
    }
    Ok(value)
}

/// Appends `text` to `out` as a quoted JSON string.
///
/// Quotes, backslashes and control characters are escaped; everything else, including
/// non-ASCII text, is written as it is. The escapes match what `serde_json` writes, so
/// a header written here is byte-identical to one written by the reference library.
pub(crate) fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn fail(&self, message: &str) -> Error {
        Error::parse(WHAT, 0, format!("{message} at byte {}", self.pos))
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<()> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.fail(&format!("expected `{}`", byte as char)))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > MAX_DEPTH {
            return Err(self.fail("nesting is too deep"));
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'0'..=b'9') => Ok(Value::Number(self.number()?)),
            Some(b'-') => Err(self.fail("negative numbers are not allowed")),
            Some(_) => Err(self.fail("unexpected character")),
            None => Err(self.fail("unexpected end of the header")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value> {
        self.expect(b'{')?;
        self.skip_whitespace();
        let mut entries = Vec::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(entries));
        }
        let mut seen = HashSet::new();
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(self.fail("expected a string key"));
            }
            let key = self.string()?;
            if !seen.insert(key.clone()) {
                return Err(self.fail(&format!("duplicate key `{key}`")));
            }
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            let value = self.value(depth + 1)?;
            entries.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Object(entries));
                }
                _ => return Err(self.fail("expected `,` or `}`")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value> {
        self.expect(b'[')?;
        self.skip_whitespace();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.value(depth + 1)?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.fail("expected `,` or `]`")),
            }
        }
    }

    fn number(&mut self) -> Result<u64> {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        let digits = &self.bytes[start..self.pos];
        if digits.len() > 1 && digits[0] == b'0' {
            return Err(self.fail("numbers must not have leading zeros"));
        }
        if matches!(self.peek(), Some(b'.' | b'e' | b'E')) {
            return Err(self.fail("fractions and exponents are not allowed"));
        }
        let mut value: u64 = 0;
        for &digit in digits {
            value = value
                .checked_mul(10)
                .and_then(|v| v.checked_add(u64::from(digit - b'0')))
                .ok_or_else(|| self.fail("number is too large"))?;
        }
        Ok(value)
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(byte) = self.peek() else {
                return Err(self.fail("unterminated string"));
            };
            self.pos += 1;
            match byte {
                b'"' => break,
                b'\\' => self.escape(&mut out)?,
                0..=0x1f => {
                    self.pos -= 1;
                    return Err(self.fail("control character in a string"));
                }
                // Every other byte, including the bytes of multi-byte characters, is
                // copied as it is. The input is valid UTF-8 and the cuts are at ASCII
                // bytes, so the result is valid UTF-8 too.
                _ => out.push(byte),
            }
        }
        String::from_utf8(out).map_err(|_| self.fail("string is not valid UTF-8"))
    }

    fn escape(&mut self, out: &mut Vec<u8>) -> Result<()> {
        let Some(byte) = self.peek() else {
            return Err(self.fail("unterminated escape"));
        };
        self.pos += 1;
        let simple = match byte {
            b'"' => Some(b'"'),
            b'\\' => Some(b'\\'),
            b'/' => Some(b'/'),
            b'b' => Some(0x08),
            b'f' => Some(0x0c),
            b'n' => Some(b'\n'),
            b'r' => Some(b'\r'),
            b't' => Some(b'\t'),
            b'u' => None,
            _ => {
                self.pos -= 1;
                return Err(self.fail("unknown escape"));
            }
        };
        if let Some(simple) = simple {
            out.push(simple);
            return Ok(());
        }
        let first = self.hex4()?;
        let code = match first {
            0xd800..=0xdbff => {
                if self.peek() != Some(b'\\') || self.bytes.get(self.pos + 1) != Some(&b'u') {
                    return Err(self.fail("unpaired surrogate in an escape"));
                }
                self.pos += 2;
                let second = self.hex4()?;
                if !(0xdc00..=0xdfff).contains(&second) {
                    return Err(self.fail("unpaired surrogate in an escape"));
                }
                0x10000 + ((u32::from(first) - 0xd800) << 10) + (u32::from(second) - 0xdc00)
            }
            0xdc00..=0xdfff => return Err(self.fail("unpaired surrogate in an escape")),
            other => u32::from(other),
        };
        let c = char::from_u32(code).ok_or_else(|| self.fail("invalid escape"))?;
        let mut buf = [0u8; 4];
        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        Ok(())
    }

    fn hex4(&mut self) -> Result<u16> {
        let mut value: u16 = 0;
        for _ in 0..4 {
            let digit = match self.peek() {
                Some(b @ b'0'..=b'9') => b - b'0',
                Some(b @ b'a'..=b'f') => b - b'a' + 10,
                Some(b @ b'A'..=b'F') => b - b'A' + 10,
                _ => return Err(self.fail("expected four hex digits after `\\u`")),
            };
            self.pos += 1;
            value = value * 16 + u16::from(digit);
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(entries: Vec<(&str, Value)>) -> Value {
        Value::Object(entries.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    fn s(text: &str) -> Value {
        Value::String(text.to_string())
    }

    #[test]
    fn parses_the_shapes_a_header_uses() {
        let got = parse(r#" {"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}, "__metadata__":{"k":"v"}} "#)
            .unwrap();
        let want = obj(vec![
            (
                "a",
                obj(vec![
                    ("dtype", s("F32")),
                    ("shape", Value::Array(vec![Value::Number(2), Value::Number(3)])),
                    ("data_offsets", Value::Array(vec![Value::Number(0), Value::Number(24)])),
                ]),
            ),
            ("__metadata__", obj(vec![("k", s("v"))])),
        ]);
        assert_eq!(got, want);
    }

    #[test]
    fn parses_empty_containers_and_the_largest_number() {
        assert_eq!(parse("{}").unwrap(), obj(vec![]));
        assert_eq!(parse("[]").unwrap(), Value::Array(vec![]));
        assert_eq!(parse("18446744073709551615").unwrap(), Value::Number(u64::MAX));
        assert_eq!(parse("0").unwrap(), Value::Number(0));
    }

    #[test]
    fn unescapes_strings_including_surrogate_pairs() {
        assert_eq!(parse(r#""a\"b\\c\/d\b\f\n\r\t""#).unwrap(), s("a\"b\\c/d\u{8}\u{c}\n\r\t"));
        assert_eq!(parse(r#""\u00e9\u0041""#).unwrap(), s("éA"));
        // U+1F600 as a surrogate pair, and raw multi-byte text passed through.
        assert_eq!(parse(r#""\ud83d\ude00 é 日本""#).unwrap(), s("\u{1f600} é 日本"));
        assert_eq!(parse(r#""\u0000""#).unwrap(), s("\u{0}"));
    }

    #[test]
    fn rejects_what_a_header_cannot_contain() {
        for bad in [
            "",
            "{",
            "{\"a\":1,}",
            "[1,]",
            "{\"a\" 1}",
            "{a:1}",
            "{\"a\":1 \"b\":2}",
            "01",
            "-1",
            "1.5",
            "1e3",
            "18446744073709551616",
            "true",
            "null",
            "\"unterminated",
            "\"bad \\q escape\"",
            "\"\\u12\"",
            "\"\\ud83d\"",
            "\"\\ud83d\\u0041\"",
            "\"\\ude00\"",
            "{} trailing",
            "{}{}",
            "\"line\nbreak\"",
        ] {
            assert!(parse(bad).is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn rejects_duplicate_keys_and_deep_nesting() {
        assert!(parse(r#"{"a":1,"a":2}"#).is_err());
        let deep = format!("{}1{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
        assert!(parse(&deep).is_err());
        let ok = format!("{}1{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(parse(&ok).is_ok());
    }

    #[test]
    fn errors_name_the_header_and_the_position() {
        let message = parse("{\"a\":x}").unwrap_err().to_string();
        assert!(message.contains("safetensors header"), "{message}");
        assert!(message.contains("byte 5"), "{message}");
    }

    #[test]
    fn written_strings_round_trip_and_match_the_reference_escapes() {
        let mut out = String::new();
        write_string(&mut out, "a\"b\\c\n\r\t\u{8}\u{c}\u{1}é\u{1f600}");
        assert_eq!(out, "\"a\\\"b\\\\c\\n\\r\\t\\b\\f\\u0001é\u{1f600}\"");
        assert_eq!(parse(&out).unwrap(), s("a\"b\\c\n\r\t\u{8}\u{c}\u{1}é\u{1f600}"));
    }
}
