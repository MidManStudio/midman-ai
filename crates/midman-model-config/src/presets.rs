// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-model-config.md, section "presets.rs"
// ============================================================================
//! Named model sizes, from a smoke test to roughly one billion parameters.

use std::fmt;
use std::str::FromStr;

use midman_foundation::Error;

use crate::config::ModelConfig;

/// A named model size.
///
/// Presets fix everything except the vocabulary size, which depends on the
/// tokenizer. All presets tie the input and output embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Preset {
    /// About 0.1M parameters. Fast enough for CI smoke tests.
    Smoke,
    /// About 1.3M parameters.
    Tiny,
    /// About 23M parameters.
    Small,
    /// About 110M parameters.
    Base,
    /// About 316M parameters, with grouped-query attention.
    Medium,
    /// About 1.0B parameters, with grouped-query attention.
    Large,
}

impl Preset {
    /// Every preset, smallest first.
    pub const ALL: [Preset; 6] =
        [Preset::Smoke, Preset::Tiny, Preset::Small, Preset::Base, Preset::Medium, Preset::Large];

    /// Lower-case name, as used in file names and on command lines.
    pub fn name(self) -> &'static str {
        match self {
            Preset::Smoke => "smoke",
            Preset::Tiny => "tiny",
            Preset::Small => "small",
            Preset::Base => "base",
            Preset::Medium => "medium",
            Preset::Large => "large",
        }
    }

    /// The vocabulary size the preset's documented parameter count assumes.
    pub fn default_vocab_size(self) -> usize {
        match self {
            Preset::Smoke => 256,
            Preset::Tiny => 4_096,
            Preset::Small | Preset::Base | Preset::Medium | Preset::Large => 32_000,
        }
    }

    /// Builds the config for this preset with the given vocabulary size.
    pub fn config(self, vocab_size: usize) -> ModelConfig {
        // (d_model, n_layers, n_heads, n_kv_heads, d_ff, max_seq_len)
        let (d_model, n_layers, n_heads, n_kv_heads, d_ff, max_seq_len) = match self {
            Preset::Smoke => (64, 2, 4, 4, 176, 128),
            Preset::Tiny => (128, 4, 4, 4, 344, 256),
            Preset::Small => (384, 6, 6, 6, 1024, 512),
            Preset::Base => (768, 12, 12, 12, 2048, 1024),
            Preset::Medium => (1024, 24, 16, 8, 2816, 2048),
            Preset::Large => (2048, 20, 16, 8, 5632, 2048),
        };
        ModelConfig {
            vocab_size,
            d_model,
            n_layers,
            n_heads,
            n_kv_heads,
            d_ff,
            max_seq_len,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    /// Builds the config using [`Preset::default_vocab_size`].
    pub fn default_config(self) -> ModelConfig {
        self.config(self.default_vocab_size())
    }
}

impl fmt::Display for Preset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Preset {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let wanted = s.trim().to_ascii_lowercase();
        Preset::ALL.into_iter().find(|p| p.name() == wanted).ok_or_else(|| {
            let valid: Vec<&str> = Preset::ALL.iter().map(|p| p.name()).collect();
            Error::parse(
                "model preset",
                0,
                format!("unknown preset `{s}`, expected one of: {}", valid.join(", ")),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_is_valid_at_its_default_vocabulary() {
        for p in Preset::ALL {
            p.default_config().validate().unwrap_or_else(|e| panic!("{p}: {e}"));
        }
    }

    #[test]
    fn parameter_counts_match_an_independent_calculation() {
        // Totals computed separately in Python from the same formula.
        let expected: [(Preset, u64); 6] = [
            (Preset::Smoke, 117_056),
            (Preset::Tiny, 1_315_968),
            (Preset::Small, 22_909_824),
            (Preset::Base, 109_529_856),
            (Preset::Medium, 315_933_696),
            (Preset::Large, 1_009_338_368),
        ];
        for (p, total) in expected {
            assert_eq!(p.default_config().param_count().unwrap(), total, "{p}");
        }
    }

    #[test]
    fn presets_grow_monotonically() {
        let counts: Vec<u64> =
            Preset::ALL.iter().map(|p| p.default_config().param_count().unwrap()).collect();
        assert!(counts.windows(2).all(|w| w[0] < w[1]), "{counts:?}");
    }

    #[test]
    fn names_round_trip_and_ignore_case() {
        for p in Preset::ALL {
            assert_eq!(p.name().parse::<Preset>().unwrap(), p);
            assert_eq!(p.to_string(), p.name());
        }
        assert_eq!("  LARGE ".parse::<Preset>().unwrap(), Preset::Large);
    }

    #[test]
    fn unknown_names_list_the_valid_ones() {
        let err = "huge".parse::<Preset>().unwrap_err().to_string();
        assert!(err.contains("huge") && err.contains("smoke") && err.contains("large"), "{err}");
    }

    #[test]
    fn a_custom_vocabulary_only_changes_the_embedding() {
        let a = Preset::Small.config(32_000).param_breakdown().unwrap();
        let b = Preset::Small.config(50_000).param_breakdown().unwrap();
        assert_eq!(b.attention, a.attention);
        assert_eq!(b.mlp, a.mlp);
        assert_eq!(b.embedding - a.embedding, 18_000 * 384);
    }
}
