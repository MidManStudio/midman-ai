// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tokenizer.md, section "special.rs"
// ============================================================================
//! The special tokens of the MidMan vocabulary.

/// Ends a document. The data pipeline writes it after every document, and it is the token
/// a model emits to say it is done.
pub const END_OF_TEXT: &str = "<|endoftext|>";

/// Fills a batch out to a fixed length where a sequence is shorter than the rest. The
/// packer never needs it; it is there for code that batches separate prompts.
pub const PAD: &str = "<|pad|>";

/// Starts the part of a document that comes before the gap in a fill-in-the-middle example.
pub const FIM_PREFIX: &str = "<|fim_prefix|>";

/// Starts the part that fills the gap in a fill-in-the-middle example.
pub const FIM_MIDDLE: &str = "<|fim_middle|>";

/// Starts the part of a document that comes after the gap in a fill-in-the-middle example.
pub const FIM_SUFFIX: &str = "<|fim_suffix|>";

/// Starts a shell command in a terminal session.
pub const COMMAND: &str = "<|command|>";

/// Starts what a command printed.
pub const OUTPUT: &str = "<|output|>";

/// Starts the exit status of a command, which is written as text after the marker.
pub const EXIT_STATUS: &str = "<|exit_status|>";

/// The agreed set, in the order it takes ids: the first one gets the first id after the
/// merges, the next the following id, and so on. [`END_OF_TEXT`] is first.
///
/// Chat and agent turn markers are not here; they wait until the format is chosen. Every
/// special token takes one id of a fixed vocabulary, so adding one later means a larger
/// embedding table or ids reserved ahead of time.
pub const SPECIAL_TOKENS: [&str; 8] =
    [END_OF_TEXT, PAD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, COMMAND, OUTPUT, EXIT_STATUS];

/// [`SPECIAL_TOKENS`] as the owned strings that `Tokenizer::new` and the trainer take.
pub fn default_special_tokens() -> Vec<String> {
    SPECIAL_TOKENS.iter().map(|text| (*text).to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PreTokenizerConfig, SpecialMode, Tokenizer, FIRST_MERGE_ID};

    #[test]
    fn the_set_is_the_agreed_eight_in_order() {
        assert_eq!(
            SPECIAL_TOKENS,
            [
                "<|endoftext|>",
                "<|pad|>",
                "<|fim_prefix|>",
                "<|fim_middle|>",
                "<|fim_suffix|>",
                "<|command|>",
                "<|output|>",
                "<|exit_status|>"
            ]
        );
        assert_eq!(default_special_tokens().len(), 8);
        assert_eq!(default_special_tokens()[0], END_OF_TEXT);
    }

    #[test]
    fn the_set_is_a_valid_special_token_list_and_takes_consecutive_ids() {
        let tokenizer =
            Tokenizer::new(PreTokenizerConfig::default(), Vec::new(), default_special_tokens())
                .unwrap();
        assert_eq!(tokenizer.vocab_size(), 256 + 8);
        for (n, text) in SPECIAL_TOKENS.iter().enumerate() {
            assert_eq!(tokenizer.special_id(text), Some(FIRST_MERGE_ID + n as u32), "{text}");
        }
    }

    #[test]
    fn the_set_survives_saving_and_loading() {
        let tokenizer =
            Tokenizer::new(PreTokenizerConfig::default(), Vec::new(), default_special_tokens())
                .unwrap();
        let loaded = Tokenizer::from_text(&tokenizer.to_text()).unwrap();
        assert_eq!(loaded.special_tokens(), tokenizer.special_tokens());
        assert_eq!(loaded.special_id(END_OF_TEXT), Some(256));
    }

    #[test]
    fn every_marker_encodes_to_its_own_id_only_when_recognized() {
        let tokenizer =
            Tokenizer::new(PreTokenizerConfig::default(), Vec::new(), default_special_tokens())
                .unwrap();
        for text in SPECIAL_TOKENS {
            let id = tokenizer.special_id(text).unwrap();
            assert_eq!(tokenizer.encode(text, SpecialMode::Recognize), [id], "{text}");
            // As ordinary text it is bytes, so data cannot inject a control token.
            let as_text = tokenizer.encode(text, SpecialMode::AsText);
            assert!(as_text.iter().all(|&t| t < 256), "{text}");
            assert_eq!(tokenizer.decode(&as_text).unwrap(), text);
        }
    }
}
