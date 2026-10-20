// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tokenizer.md, section "tests/reference_oracle.rs"
// ============================================================================
//! Tests against an independent implementation.
//!
//! The fixtures in `tests/fixtures/` were written by `reference.py`, which has a
//! regular-expression pre-tokenizer and a textbook BPE trainer and shares no code
//! with this crate. Agreement with it is the strongest evidence that the Rust
//! pre-tokenizer, trainer and encoder do what `docs/midman-tokenizer.md` says.

use midman_tokenizer::{
    PreTokenizerConfig, SpecialMode, Trainer, TrainerConfig, WhitespaceRule, WordPrefix,
};

fn unhex(hex: &str) -> Vec<u8> {
    assert_eq!(hex.len() % 2, 0, "odd number of hex digits");
    (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect()
}

/// Reads a setting written as `ansi,underscore,prefix,digits,punct_newlines,whitespace`.
fn parse_config(text: &str) -> PreTokenizerConfig {
    let f: Vec<&str> = text.split(',').collect();
    assert_eq!(f.len(), 6, "{text}");
    PreTokenizerConfig {
        ansi: f[0] == "1",
        underscore_in_words: f[1] == "1",
        word_prefix: match f[2] {
            "space" => WordPrefix::Space,
            "any" => WordPrefix::AnyOne,
            other => panic!("unknown prefix {other}"),
        },
        digit_run: f[3].parse().unwrap(),
        punct_newlines: f[4] == "1",
        whitespace: match f[5] {
            "gpt2" => WhitespaceRule::Gpt2,
            "cl100k" => WhitespaceRule::Cl100k,
            other => panic!("unknown whitespace rule {other}"),
        },
    }
}

#[test]
fn the_pre_tokenizer_cuts_every_golden_case_exactly_as_the_reference_does() {
    let golden = include_str!("fixtures/pretokenize_golden.txt");
    let mut cases = 0;
    let mut settings = std::collections::HashSet::new();
    for (number, line) in golden.lines().enumerate() {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 3, "line {}", number + 1);
        let config = parse_config(fields[0]);
        settings.insert(fields[0]);
        let input = unhex(fields[1]);
        let want: Vec<Vec<u8>> = if fields[2].is_empty() {
            Vec::new()
        } else {
            fields[2].split('|').map(unhex).collect()
        };
        let got: Vec<Vec<u8>> = config.pieces(&input).map(<[u8]>::to_vec).collect();
        assert_eq!(got, want, "line {}: setting {} input {:?}", number + 1, fields[0], fields[1]);
        cases += 1;
    }
    assert_eq!(cases, 410);
    assert_eq!(settings.len(), 8);
}

struct BpeGolden {
    config: PreTokenizerConfig,
    min_frequency: u64,
    requested_merges: usize,
    merges: Vec<(u32, u32)>,
    encodings: Vec<(Vec<u8>, Vec<u32>)>,
}

fn read_bpe_golden(text: &str) -> BpeGolden {
    let mut lines = text.lines();
    let mut field = |name: &str| -> String {
        let line = lines.next().unwrap();
        line.strip_prefix(name).and_then(|rest| rest.strip_prefix(' ')).unwrap().to_string()
    };
    let config = parse_config(&field("config"));
    let min_frequency = field("min_frequency").parse().unwrap();
    let requested_merges = field("requested_merges").parse().unwrap();
    let count: usize = field("merges").parse().unwrap();
    let mut merges = Vec::new();
    for _ in 0..count {
        let line = lines.next().unwrap();
        let (a, b) = line.split_once(' ').unwrap();
        merges.push((a.parse().unwrap(), b.parse().unwrap()));
    }
    assert_eq!(lines.next(), Some("---"));
    let mut encodings = Vec::new();
    for line in lines {
        let (hex, ids) = line.split_once('\t').unwrap();
        let ids = if ids.is_empty() {
            Vec::new()
        } else {
            ids.split(' ').map(|n| n.parse().unwrap()).collect()
        };
        encodings.push((unhex(hex), ids));
    }
    BpeGolden { config, min_frequency, requested_merges, merges, encodings }
}

/// The corpus documents, separated by a line of five equals signs. Escape characters
/// are written as the four characters `\x1b` in the file and restored here.
fn corpus_documents() -> Vec<String> {
    let text = include_str!("fixtures/bpe_corpus.txt").replace("\\x1b", "\u{1b}");
    text.split("\n=====\n").map(str::to_string).collect()
}

fn check_against(golden_text: &str) {
    let golden = read_bpe_golden(golden_text);
    let mut config = TrainerConfig::new(256 + golden.requested_merges);
    config.pretokenizer = golden.config;
    config.min_frequency = golden.min_frequency;
    let mut trainer = Trainer::new(config).unwrap();
    let docs = corpus_documents();
    assert_eq!(docs.len(), 7);
    for doc in &docs {
        trainer.add(doc.as_bytes());
    }
    let (tokenizer, stats) = trainer.finish().unwrap();
    assert_eq!(tokenizer.merges().len(), golden.merges.len());
    assert_eq!(stats.merges, golden.merges.len());
    assert_eq!(tokenizer.merges(), golden.merges.as_slice(), "the learned merges differ");
    assert!(!golden.encodings.is_empty());
    for (input, want) in &golden.encodings {
        let got = tokenizer.encode_bytes(input, SpecialMode::AsText);
        assert_eq!(&got, want, "encoding of {:?}", String::from_utf8_lossy(input));
        assert_eq!(&tokenizer.decode_bytes(&got).unwrap(), input);
    }
}

#[test]
fn training_on_the_corpus_learns_the_same_merges_and_encodes_the_same_ids_without_ansi() {
    check_against(include_str!("fixtures/bpe_golden_a.txt"));
}

#[test]
fn training_on_the_corpus_learns_the_same_merges_and_encodes_the_same_ids_for_code_and_shell() {
    check_against(include_str!("fixtures/bpe_golden_b.txt"));
}

#[test]
fn the_two_settings_really_learn_different_vocabularies() {
    let a = read_bpe_golden(include_str!("fixtures/bpe_golden_a.txt"));
    let b = read_bpe_golden(include_str!("fixtures/bpe_golden_b.txt"));
    assert_ne!(a.merges, b.merges);
    assert_eq!(a.merges.len(), 300);
    assert_eq!(b.merges.len(), 300);
}
