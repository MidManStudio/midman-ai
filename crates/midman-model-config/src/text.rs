// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-model-config.md, section "text.rs"
// ============================================================================
//! Reading and writing a [`ModelConfig`] as a flat, TOML-compatible text file.
//!
//! Only the TOML subset the config needs is supported: `key = value` lines,
//! integers (with optional `_` separators), floats, booleans, blank lines and
//! `#` comments. There are no tables and no strings. Files are strict: every key
//! must be present, unknown or duplicate keys are errors, and the schema version
//! must match, so a config never silently falls back to a default.

use midman_foundation::{check_schema_version, Error, Result};

use crate::config::{ModelConfig, CONFIG_SCHEMA_VERSION};

const WHAT: &str = "model config";

const REQUIRED_KEYS: [&str; 11] = [
    "schema_version",
    "vocab_size",
    "d_model",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "d_ff",
    "max_seq_len",
    "rope_theta",
    "norm_eps",
    "tie_embeddings",
];

fn parse_usize(line: usize, key: &str, value: &str) -> Result<usize> {
    value.replace('_', "").parse::<usize>().map_err(|_| {
        Error::parse(WHAT, line, format!("`{key}` must be a non-negative integer, got `{value}`"))
    })
}

fn parse_u32(line: usize, key: &str, value: &str) -> Result<u32> {
    value.replace('_', "").parse::<u32>().map_err(|_| {
        Error::parse(WHAT, line, format!("`{key}` must be a non-negative integer, got `{value}`"))
    })
}

fn parse_f32(line: usize, key: &str, value: &str) -> Result<f32> {
    value
        .replace('_', "")
        .parse::<f32>()
        .map_err(|_| Error::parse(WHAT, line, format!("`{key}` must be a number, got `{value}`")))
}

fn parse_bool(line: usize, key: &str, value: &str) -> Result<bool> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(Error::parse(WHAT, line, format!("`{key}` must be true or false, got `{value}`"))),
    }
}

impl ModelConfig {
    /// Writes the config as text that [`ModelConfig::from_toml_str`] reads back
    /// to an equal value.
    pub fn to_toml_string(&self) -> String {
        format!(
            "schema_version = {CONFIG_SCHEMA_VERSION}\n\
             vocab_size = {}\n\
             d_model = {}\n\
             n_layers = {}\n\
             n_heads = {}\n\
             n_kv_heads = {}\n\
             d_ff = {}\n\
             max_seq_len = {}\n\
             rope_theta = {:?}\n\
             norm_eps = {:?}\n\
             tie_embeddings = {}\n",
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.d_ff,
            self.max_seq_len,
            self.rope_theta,
            self.norm_eps,
            self.tie_embeddings,
        )
    }

    /// Parses a config from text and validates it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Parse`] for malformed, unknown or duplicate lines,
    /// [`Error::UnsupportedVersion`] if `schema_version` is not
    /// [`CONFIG_SCHEMA_VERSION`], and [`Error::Config`] for missing keys or
    /// values that fail [`ModelConfig::validate`].
    pub fn from_toml_str(text: &str) -> Result<ModelConfig> {
        let mut entries: Vec<(usize, &str, &str)> = Vec::new();
        for (idx, raw) in text.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                return Err(Error::parse(WHAT, line_no, "expected `key = value`"));
            };
            entries.push((line_no, key.trim(), value.trim()));
        }

        // Check the version first so a file from a newer format reports that,
        // instead of failing on a key this build does not know.
        let Some(&(line_no, key, value)) = entries.iter().find(|(_, k, _)| *k == "schema_version")
        else {
            return Err(Error::config("schema_version", "missing from the config"));
        };
        check_schema_version(WHAT, parse_u32(line_no, key, value)?, CONFIG_SCHEMA_VERSION)?;

        let mut cfg = ModelConfig {
            vocab_size: 0,
            d_model: 0,
            n_layers: 0,
            n_heads: 0,
            n_kv_heads: 0,
            d_ff: 0,
            max_seq_len: 0,
            rope_theta: 0.0,
            norm_eps: 0.0,
            tie_embeddings: false,
        };
        let mut seen: Vec<&str> = Vec::new();
        for &(line_no, key, value) in &entries {
            if seen.contains(&key) {
                return Err(Error::parse(WHAT, line_no, format!("duplicate key `{key}`")));
            }
            seen.push(key);
            match key {
                "schema_version" => {}
                "vocab_size" => cfg.vocab_size = parse_usize(line_no, key, value)?,
                "d_model" => cfg.d_model = parse_usize(line_no, key, value)?,
                "n_layers" => cfg.n_layers = parse_usize(line_no, key, value)?,
                "n_heads" => cfg.n_heads = parse_usize(line_no, key, value)?,
                "n_kv_heads" => cfg.n_kv_heads = parse_usize(line_no, key, value)?,
                "d_ff" => cfg.d_ff = parse_usize(line_no, key, value)?,
                "max_seq_len" => cfg.max_seq_len = parse_usize(line_no, key, value)?,
                "rope_theta" => cfg.rope_theta = parse_f32(line_no, key, value)?,
                "norm_eps" => cfg.norm_eps = parse_f32(line_no, key, value)?,
                "tie_embeddings" => cfg.tie_embeddings = parse_bool(line_no, key, value)?,
                _ => return Err(Error::parse(WHAT, line_no, format!("unknown key `{key}`"))),
            }
        }

        let missing: Vec<&str> =
            REQUIRED_KEYS.iter().copied().filter(|k| !seen.contains(k)).collect();
        if !missing.is_empty() {
            return Err(Error::config(missing.join(", "), "missing from the config"));
        }
        cfg.validate()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ModelConfig {
        ModelConfig {
            vocab_size: 32_000,
            d_model: 384,
            n_layers: 6,
            n_heads: 6,
            n_kv_heads: 3,
            d_ff: 1024,
            max_seq_len: 512,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    const SAMPLE_TEXT: &str = "schema_version = 1\n\
        vocab_size = 32000\n\
        d_model = 384\n\
        n_layers = 6\n\
        n_heads = 6\n\
        n_kv_heads = 3\n\
        d_ff = 1024\n\
        max_seq_len = 512\n\
        rope_theta = 10000.0\n\
        norm_eps = 1e-5\n\
        tie_embeddings = true\n";

    #[test]
    fn writes_the_documented_text_format() {
        assert_eq!(sample().to_toml_string(), SAMPLE_TEXT);
    }

    #[test]
    fn round_trips_exactly() {
        let cfg = sample();
        assert_eq!(ModelConfig::from_toml_str(&cfg.to_toml_string()).unwrap(), cfg);
    }

    #[test]
    fn round_trips_awkward_floats_bit_for_bit() {
        let mut cfg = sample();
        cfg.rope_theta = 500_000.0;
        cfg.norm_eps = 1.1920929e-7;
        let back = ModelConfig::from_toml_str(&cfg.to_toml_string()).unwrap();
        assert_eq!(back.rope_theta.to_bits(), cfg.rope_theta.to_bits());
        assert_eq!(back.norm_eps.to_bits(), cfg.norm_eps.to_bits());
    }

    #[test]
    fn accepts_comments_blank_lines_and_digit_separators() {
        let text = format!(
            "# a comment\n\n{}  # trailing comment\n",
            SAMPLE_TEXT.replace("32000", "32_000")
        );
        assert_eq!(ModelConfig::from_toml_str(&text).unwrap(), sample());
    }

    fn parse_error_line(text: &str) -> (usize, String) {
        match ModelConfig::from_toml_str(text).unwrap_err() {
            Error::Parse { line, message, .. } => (line, message),
            other => panic!("expected a parse error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_a_line_without_an_equals_sign() {
        let (line, msg) = parse_error_line("schema_version = 1\nvocab_size 5\n");
        assert_eq!(line, 2);
        assert!(msg.contains("key = value"), "{msg}");
    }

    #[test]
    fn rejects_unknown_keys_with_the_line_number() {
        let text = format!("{SAMPLE_TEXT}extra_key = 3\n");
        let (line, msg) = parse_error_line(&text);
        assert_eq!(line, 12);
        assert!(msg.contains("extra_key"), "{msg}");
    }

    #[test]
    fn rejects_duplicate_keys() {
        let text = format!("{SAMPLE_TEXT}d_model = 128\n");
        let (line, msg) = parse_error_line(&text);
        assert_eq!(line, 12);
        assert!(msg.contains("duplicate"), "{msg}");
    }

    #[test]
    fn rejects_values_of_the_wrong_type() {
        let (_, msg) = parse_error_line(&SAMPLE_TEXT.replace("= 384", "= -384"));
        assert!(msg.contains("d_model"), "{msg}");
        let (_, msg) = parse_error_line(&SAMPLE_TEXT.replace("= 10000.0", "= fast"));
        assert!(msg.contains("rope_theta"), "{msg}");
        let (_, msg) = parse_error_line(&SAMPLE_TEXT.replace("= true", "= yes"));
        assert!(msg.contains("tie_embeddings"), "{msg}");
    }

    #[test]
    fn names_every_missing_key() {
        let text = "schema_version = 1\nvocab_size = 10\n";
        match ModelConfig::from_toml_str(text).unwrap_err() {
            Error::Config { field, reason } => {
                assert!(field.contains("d_model") && field.contains("tie_embeddings"), "{field}");
                assert!(reason.contains("missing"), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn requires_a_schema_version() {
        let text = SAMPLE_TEXT.replace("schema_version = 1\n", "");
        match ModelConfig::from_toml_str(&text).unwrap_err() {
            Error::Config { field, .. } => assert_eq!(field, "schema_version"),
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn a_newer_schema_reports_the_version_not_unknown_keys() {
        let text = format!(
            "{}new_future_key = 1\n",
            SAMPLE_TEXT.replace("schema_version = 1", "schema_version = 2")
        );
        assert_eq!(
            ModelConfig::from_toml_str(&text).unwrap_err(),
            Error::UnsupportedVersion { what: "model config".into(), found: 2, supported: 1 }
        );
    }

    #[test]
    fn parsed_values_are_validated() {
        let text = SAMPLE_TEXT.replace("n_heads = 6", "n_heads = 5");
        match ModelConfig::from_toml_str(&text).unwrap_err() {
            Error::Config { field, .. } => assert_eq!(field, "n_heads"),
            other => panic!("unexpected error {other:?}"),
        }
    }
}
