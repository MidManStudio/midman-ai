// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-api.md, section "json.rs"
// ============================================================================
//! A small JSON writer. It only writes; nothing in this crate reads JSON.

/// Builds a JSON document one value at a time, tracking commas for the caller.
///
/// Numbers that JSON cannot represent (NaN and infinities) are written as
/// `null`, so the output always parses.
#[derive(Debug, Default)]
pub(crate) struct JsonWriter {
    out: String,
    /// One entry per open container: whether it already holds a value.
    has_value: Vec<bool>,
    /// A key was just written, so the next value follows a colon.
    after_key: bool,
}

impl JsonWriter {
    pub(crate) fn new() -> JsonWriter {
        JsonWriter::default()
    }

    /// Finishes the document and returns it.
    pub(crate) fn finish(self) -> String {
        debug_assert!(self.has_value.is_empty(), "unclosed JSON container");
        self.out
    }

    fn before_value(&mut self) {
        if self.after_key {
            self.after_key = false;
        } else if let Some(has) = self.has_value.last_mut() {
            if *has {
                self.out.push(',');
            }
            *has = true;
        }
    }

    pub(crate) fn begin_object(&mut self) -> &mut Self {
        self.before_value();
        self.out.push('{');
        self.has_value.push(false);
        self
    }

    pub(crate) fn end_object(&mut self) -> &mut Self {
        self.has_value.pop();
        self.out.push('}');
        self
    }

    pub(crate) fn begin_array(&mut self) -> &mut Self {
        self.before_value();
        self.out.push('[');
        self.has_value.push(false);
        self
    }

    pub(crate) fn end_array(&mut self) -> &mut Self {
        self.has_value.pop();
        self.out.push(']');
        self
    }

    /// Writes an object key; the next call writes its value.
    pub(crate) fn key(&mut self, name: &str) -> &mut Self {
        if let Some(has) = self.has_value.last_mut() {
            if *has {
                self.out.push(',');
            }
            *has = true;
        }
        escape_into(&mut self.out, name);
        self.out.push(':');
        self.after_key = true;
        self
    }

    pub(crate) fn string(&mut self, value: &str) -> &mut Self {
        self.before_value();
        escape_into(&mut self.out, value);
        self
    }

    pub(crate) fn number(&mut self, value: f64) -> &mut Self {
        self.before_value();
        if value.is_finite() {
            self.out.push_str(&value.to_string());
        } else {
            self.out.push_str("null");
        }
        self
    }

    pub(crate) fn integer(&mut self, value: u64) -> &mut Self {
        self.before_value();
        self.out.push_str(&value.to_string());
        self
    }

    pub(crate) fn null(&mut self) -> &mut Self {
        self.before_value();
        self.out.push_str("null");
        self
    }

    /// A number, or `null` when there is none.
    pub(crate) fn optional_number(&mut self, value: Option<f64>) -> &mut Self {
        match value {
            Some(v) => self.number(v),
            None => self.null(),
        }
    }

    pub(crate) fn boolean(&mut self, value: bool) -> &mut Self {
        self.before_value();
        self.out.push_str(if value { "true" } else { "false" });
        self
    }
}

/// Appends `value` as a quoted JSON string.
fn escape_into(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_values_get_their_commas_and_colons() {
        let mut w = JsonWriter::new();
        w.begin_object();
        w.key("a").integer(1);
        w.key("list").begin_array();
        w.number(1.5).string("x").boolean(true);
        w.begin_object().key("k").number(2.0).end_object();
        w.end_array();
        w.key("empty").begin_array().end_array();
        w.end_object();
        assert_eq!(w.finish(), r#"{"a":1,"list":[1.5,"x",true,{"k":2}],"empty":[]}"#);
    }

    #[test]
    fn strings_are_escaped() {
        let mut w = JsonWriter::new();
        w.begin_array().string("a\"b\\c\nd\te\u{1}").end_array();
        assert_eq!(w.finish(), "[\"a\\\"b\\\\c\\nd\\te\\u0001\"]");
    }

    #[test]
    fn values_json_cannot_hold_become_null() {
        let mut w = JsonWriter::new();
        w.begin_array()
            .number(f64::NAN)
            .number(f64::INFINITY)
            .number(f64::NEG_INFINITY)
            .end_array();
        assert_eq!(w.finish(), "[null,null,null]");
    }

    #[test]
    fn a_missing_number_is_null() {
        let mut w = JsonWriter::new();
        w.begin_array().optional_number(Some(1.0)).optional_number(None).null().end_array();
        assert_eq!(w.finish(), "[1,null,null]");
    }

    #[test]
    fn a_top_level_scalar_is_allowed() {
        let mut w = JsonWriter::new();
        w.integer(7);
        assert_eq!(w.finish(), "7");
    }
}
