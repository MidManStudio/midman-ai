// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tokenizer.md, section "tokenizer.rs"
// ============================================================================
//! The tokenizer: a byte-level BPE vocabulary with encoding, decoding, special tokens
//! and a saved text format.
//!
//! Token ids are laid out in three blocks. Ids `0..=255` are the raw bytes, so an id
//! below 256 is its own byte. Then come the merges, one id each in the order they were
//! learned: merge `r` joins two earlier tokens into token `256 + r`. The special tokens
//! come last. Because every byte has an id, any byte string can be encoded and decoded
//! exactly, whether or not it is valid UTF-8.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use midman_foundation::{Error, Result};

use crate::pretokenize::{PreTokenizerConfig, WhitespaceRule, WordPrefix};

/// The version of the saved text format this build writes and reads.
pub const FORMAT_VERSION: u32 = 1;

/// The first id that is a merge instead of a byte.
pub const FIRST_MERGE_ID: u32 = 256;

/// What the errors say was being read.
const WHAT: &str = "tokenizer file";

/// The first word of a saved tokenizer.
const MAGIC: &str = "midman-tokenizer";

/// Whether special-token text in the input is recognized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialMode {
    /// Special-token text is ordinary text and is encoded byte by byte. Use this for
    /// training data and anything else from outside, so that text that happens to read
    /// `<|endoftext|>` cannot inject a control token.
    AsText,
    /// Special-token text becomes the token's id. Use this for text the program
    /// builds itself, such as a prompt template.
    Recognize,
}

/// A trained vocabulary and the rules for using it.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    config: PreTokenizerConfig,
    merges: Vec<(u32, u32)>,
    /// The merge that makes each pair, as the id of the token it produces.
    merge_ids: HashMap<(u32, u32), u32>,
    /// The bytes of every non-special token, indexed by id.
    token_bytes: Vec<Vec<u8>>,
    specials: Vec<String>,
    /// Indices into `specials`, longest text first, so the longest match wins.
    special_order: Vec<usize>,
    /// Which bytes some special token starts with.
    special_first: [bool; 256],
}

impl Tokenizer {
    /// Builds a tokenizer from merges in rank order and a list of special tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if a merge uses an id that does not exist yet, if two merges
    /// join the same pair, or if a special token is empty, repeated, has leading or
    /// trailing whitespace or contains a line break.
    pub fn new(
        config: PreTokenizerConfig,
        merges: Vec<(u32, u32)>,
        specials: Vec<String>,
    ) -> Result<Tokenizer> {
        if u32::try_from(merges.len())
            .ok()
            .and_then(|n| n.checked_add(FIRST_MERGE_ID + 1))
            .is_none()
        {
            return Err(Error::invalid("there are too many merges for 32-bit token ids"));
        }
        validate_specials(&specials)?;
        let mut token_bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut merge_ids = HashMap::with_capacity(merges.len());
        for (rank, &(left, right)) in merges.iter().enumerate() {
            let id = FIRST_MERGE_ID + rank as u32;
            if left >= id || right >= id {
                return Err(Error::invalid(format!(
                    "merge {rank} joins tokens {left} and {right}, but only tokens below {id} exist yet"
                )));
            }
            if merge_ids.insert((left, right), id).is_some() {
                return Err(Error::invalid(format!(
                    "merge {rank} repeats the pair ({left}, {right})"
                )));
            }
            let mut bytes = token_bytes[left as usize].clone();
            bytes.extend_from_slice(&token_bytes[right as usize]);
            token_bytes.push(bytes);
        }
        let mut special_order: Vec<usize> = (0..specials.len()).collect();
        special_order.sort_by_key(|&i| std::cmp::Reverse(specials[i].len()));
        let mut special_first = [false; 256];
        for text in &specials {
            special_first[usize::from(text.as_bytes()[0])] = true;
        }
        Ok(Tokenizer {
            config,
            merges,
            merge_ids,
            token_bytes,
            specials,
            special_order,
            special_first,
        })
    }

    /// The tokenizer with no merges and no special tokens: one token per byte, a
    /// vocabulary of 256, which is what the Smoke preset uses.
    pub fn bytes_only() -> Tokenizer {
        match Tokenizer::new(PreTokenizerConfig::default(), Vec::new(), Vec::new()) {
            Ok(tokenizer) => tokenizer,
            Err(_) => unreachable!("an empty merge list and no specials are always valid"),
        }
    }

    /// The number of tokens: bytes, merges and special tokens.
    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len() + self.specials.len()
    }

    /// The merges in rank order. Merge `r` makes token `256 + r`.
    pub fn merges(&self) -> &[(u32, u32)] {
        &self.merges
    }

    /// The special tokens, in id order. The first one's id is `256 + merges().len()`.
    pub fn special_tokens(&self) -> &[String] {
        &self.specials
    }

    /// The id of the special token with this exact text.
    pub fn special_id(&self, text: &str) -> Option<u32> {
        let base = self.token_bytes.len();
        self.specials.iter().position(|s| s == text).map(|i| (base + i) as u32)
    }

    /// The pre-tokenizer settings this vocabulary was trained with.
    pub fn pretokenizer(&self) -> &PreTokenizerConfig {
        &self.config
    }

    /// The bytes a token stands for. For a special token these are the bytes of its
    /// text. `None` if the id is out of range.
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        let id = id as usize;
        if let Some(bytes) = self.token_bytes.get(id) {
            return Some(bytes);
        }
        self.specials.get(id - self.token_bytes.len()).map(|s| s.as_bytes())
    }

    /// Encodes text. See [`Tokenizer::encode_bytes`].
    pub fn encode(&self, text: &str, mode: SpecialMode) -> Vec<u32> {
        self.encode_bytes(text.as_bytes(), mode)
    }

    /// Encodes any byte string. Decoding the result gives the same bytes back, and no
    /// piece the pre-tokenizer cuts is ever joined across its boundary.
    pub fn encode_bytes(&self, text: &[u8], mode: SpecialMode) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        if mode == SpecialMode::AsText || self.specials.is_empty() {
            self.encode_plain(text, &mut out);
            return out;
        }
        let mut start = 0;
        while let Some((at, len, index)) = self.find_special(text, start) {
            self.encode_plain(&text[start..at], &mut out);
            out.push((self.token_bytes.len() + index) as u32);
            start = at + len;
        }
        self.encode_plain(&text[start..], &mut out);
        out
    }

    /// The first special token at or after `from`, as `(position, length, index)`. At a
    /// position where several match, the longest one.
    fn find_special(&self, text: &[u8], from: usize) -> Option<(usize, usize, usize)> {
        for at in from..text.len() {
            if !self.special_first[usize::from(text[at])] {
                continue;
            }
            for &index in &self.special_order {
                let special = self.specials[index].as_bytes();
                if text[at..].starts_with(special) {
                    return Some((at, special.len(), index));
                }
            }
        }
        None
    }

    fn encode_plain(&self, text: &[u8], out: &mut Vec<u32>) {
        for piece in self.config.pieces(text) {
            self.encode_piece(piece, out);
        }
    }

    /// Applies the merges to one piece: repeatedly take the earliest-learned merge that
    /// applies anywhere and apply it to every non-overlapping occurrence, left to
    /// right. This is exactly the order the trainer applied them in.
    fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        let mut ids: Vec<u32> = piece.iter().map(|&b| u32::from(b)).collect();
        while ids.len() > 1 {
            let mut best: Option<u32> = None;
            for pair in ids.windows(2) {
                if let Some(&id) = self.merge_ids.get(&(pair[0], pair[1])) {
                    if !matches!(best, Some(b) if b <= id) {
                        best = Some(id);
                    }
                }
            }
            let Some(id) = best else { break };
            let pair = self.merges[(id - FIRST_MERGE_ID) as usize];
            let (mut read, mut write) = (0, 0);
            while read < ids.len() {
                if read + 1 < ids.len() && (ids[read], ids[read + 1]) == pair {
                    ids[write] = id;
                    read += 2;
                } else {
                    ids[write] = ids[read];
                    read += 1;
                }
                write += 1;
            }
            ids.truncate(write);
        }
        out.extend_from_slice(&ids);
    }

    /// Decodes token ids to bytes. Special tokens decode to their text.
    ///
    /// # Errors
    ///
    /// Returns an error naming the first id that is not in the vocabulary.
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(ids.len() * 3);
        for &id in ids {
            match self.token_bytes(id) {
                Some(bytes) => out.extend_from_slice(bytes),
                None => return Err(Error::index("token id", id as usize, self.vocab_size())),
            }
        }
        Ok(out)
    }

    /// Decodes token ids to text, replacing any bytes that are not valid UTF-8 with
    /// U+FFFD. Use [`Tokenizer::decode_bytes`] where the exact bytes matter.
    ///
    /// # Errors
    ///
    /// Returns an error naming the first id that is not in the vocabulary.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?).into_owned())
    }

    /// Writes the tokenizer in its saved text format.
    ///
    /// The first line names the format and version, the second holds the pre-tokenizer
    /// settings, then come one `special` line per special token and one `merge` line
    /// per merge, in id order.
    pub fn to_text(&self) -> String {
        let c = &self.config;
        let mut out = format!("{MAGIC} {FORMAT_VERSION}\n");
        out.push_str(&format!(
            "pretokenizer ansi={} underscore={} prefix={} digits={} punct_newlines={} whitespace={}\n",
            u8::from(c.ansi),
            u8::from(c.underscore_in_words),
            match c.word_prefix {
                WordPrefix::Space => "space",
                WordPrefix::AnyOne => "any",
            },
            c.digit_run,
            u8::from(c.punct_newlines),
            match c.whitespace {
                WhitespaceRule::Gpt2 => "gpt2",
                WhitespaceRule::Cl100k => "cl100k",
            },
        ));
        for special in &self.specials {
            out.push_str("special ");
            out.push_str(special);
            out.push('\n');
        }
        for (left, right) in &self.merges {
            out.push_str(&format!("merge {left} {right}\n"));
        }
        out
    }

    /// Reads a tokenizer from its saved text format.
    ///
    /// # Errors
    ///
    /// Returns an error naming the line if the text is not a tokenizer file: a wrong
    /// header, an unknown version, a missing, repeated or unknown setting, lines out of
    /// order, a blank line, or a number that does not parse. Merges and special tokens
    /// are then checked as for [`Tokenizer::new`].
    pub fn from_text(text: &str) -> Result<Tokenizer> {
        let mut lines = text.lines().enumerate().map(|(i, line)| (i + 1, line));
        let (_, header) = lines.next().ok_or_else(|| Error::parse(WHAT, 0, "the file is empty"))?;
        let version =
            header.strip_prefix(MAGIC).and_then(|rest| rest.strip_prefix(' ')).ok_or_else(
                || Error::parse(WHAT, 1, format!("the first line must start with `{MAGIC} `")),
            )?;
        let version: u32 = version
            .parse()
            .map_err(|_| Error::parse(WHAT, 1, format!("`{version}` is not a version number")))?;
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion {
                what: "tokenizer file".to_string(),
                found: version,
                supported: FORMAT_VERSION,
            });
        }

        let (n, settings) = lines
            .next()
            .ok_or_else(|| Error::parse(WHAT, 2, "the pretokenizer line is missing"))?;
        let config = parse_pretokenizer(n, settings)?;

        let mut specials = Vec::new();
        let mut merges = Vec::new();
        for (n, line) in lines {
            if let Some(special) = line.strip_prefix("special ") {
                if !merges.is_empty() {
                    return Err(Error::parse(WHAT, n, "special tokens must come before merges"));
                }
                specials.push(special.to_string());
            } else if let Some(pair) = line.strip_prefix("merge ") {
                let mut numbers = pair.split(' ').map(str::parse::<u32>);
                match (numbers.next(), numbers.next(), numbers.next()) {
                    (Some(Ok(left)), Some(Ok(right)), None) => merges.push((left, right)),
                    _ => {
                        return Err(Error::parse(
                            WHAT,
                            n,
                            "a merge line is `merge <left id> <right id>`",
                        ));
                    }
                }
            } else {
                return Err(Error::parse(WHAT, n, "expected a `special` or a `merge` line"));
            }
        }
        Tokenizer::new(config, merges, specials)
    }

    /// Writes the saved text format to `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        fs::write(path, self.to_text())
            .map_err(|err| Error::Io(format!("{}: {err}", path.display())))
    }

    /// Reads a tokenizer from a file in the saved text format.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is not a tokenizer file.
    pub fn load(path: impl AsRef<Path>) -> Result<Tokenizer> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|err| Error::Io(format!("{}: {err}", path.display())))?;
        Tokenizer::from_text(&text)
    }
}

/// Checks a list of special tokens against the rules the saved format needs.
pub(crate) fn validate_specials(specials: &[String]) -> Result<()> {
    for (i, text) in specials.iter().enumerate() {
        if text.is_empty() {
            return Err(Error::invalid(format!("special token {i} is empty")));
        }
        if text.contains(['\n', '\r']) {
            return Err(Error::invalid(format!("special token {i} contains a line break")));
        }
        if text.trim() != text {
            return Err(Error::invalid(format!(
                "special token `{text}` has leading or trailing whitespace"
            )));
        }
        if specials[..i].contains(text) {
            return Err(Error::invalid(format!("special token `{text}` is listed twice")));
        }
    }
    Ok(())
}

fn parse_pretokenizer(line_number: usize, line: &str) -> Result<PreTokenizerConfig> {
    let fail = |message: String| Error::parse(WHAT, line_number, message);
    let rest = line
        .strip_prefix("pretokenizer")
        .ok_or_else(|| fail("the second line must start with `pretokenizer`".to_string()))?;
    let mut config = PreTokenizerConfig::default();
    let mut seen = [false; 6];
    for setting in rest.split_whitespace() {
        let (key, value) = setting
            .split_once('=')
            .ok_or_else(|| fail(format!("`{setting}` is not `key=value`")))?;
        let flag = |value: &str| match value {
            "0" => Ok(false),
            "1" => Ok(true),
            other => Err(fail(format!("`{key}` must be 0 or 1, not `{other}`"))),
        };
        let slot = match key {
            "ansi" => {
                config.ansi = flag(value)?;
                0
            }
            "underscore" => {
                config.underscore_in_words = flag(value)?;
                1
            }
            "prefix" => {
                config.word_prefix = match value {
                    "space" => WordPrefix::Space,
                    "any" => WordPrefix::AnyOne,
                    other => {
                        return Err(fail(format!("`prefix` must be space or any, not `{other}`")))
                    }
                };
                2
            }
            "digits" => {
                config.digit_run = value.parse().map_err(|_| {
                    fail(format!("`digits` must be a number from 0 to 255, not `{value}`"))
                })?;
                3
            }
            "punct_newlines" => {
                config.punct_newlines = flag(value)?;
                4
            }
            "whitespace" => {
                config.whitespace = match value {
                    "gpt2" => WhitespaceRule::Gpt2,
                    "cl100k" => WhitespaceRule::Cl100k,
                    other => {
                        return Err(fail(format!(
                            "`whitespace` must be gpt2 or cl100k, not `{other}`"
                        )));
                    }
                };
                5
            }
            other => return Err(fail(format!("unknown pretokenizer setting `{other}`"))),
        };
        if seen[slot] {
            return Err(fail(format!("`{key}` is set twice")));
        }
        seen[slot] = true;
    }
    if let Some(missing) = seen.iter().position(|&s| !s) {
        let names = ["ansi", "underscore", "prefix", "digits", "punct_newlines", "whitespace"];
        return Err(fail(format!("the setting `{}` is missing", names[missing])));
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_foundation::Rng;

    fn tok(merges: &[(u32, u32)], specials: &[&str]) -> Tokenizer {
        let specials = specials.iter().map(|s| s.to_string()).collect();
        Tokenizer::new(PreTokenizerConfig::gpt2_like(), merges.to_vec(), specials).unwrap()
    }

    const T: u32 = b't' as u32;
    const H: u32 = b'h' as u32;
    const E: u32 = b'e' as u32;
    const A: u32 = b'a' as u32;

    #[test]
    fn a_bytes_only_tokenizer_maps_every_byte_to_its_own_id() {
        let t = Tokenizer::bytes_only();
        assert_eq!(t.vocab_size(), 256);
        assert_eq!(t.encode("hé", SpecialMode::AsText), [104, 195, 169]);
        let all: Vec<u8> = (0..=255).collect();
        let ids = t.encode_bytes(&all, SpecialMode::AsText);
        assert_eq!(ids, (0..=255).collect::<Vec<u32>>());
        assert_eq!(t.decode_bytes(&ids).unwrap(), all);
    }

    #[test]
    fn merges_apply_in_rank_order_and_need_not_be_the_longest_match() {
        // 256 = th, 257 = the, 258 = ha
        let t = tok(&[(T, H), (256, E), (H, A)], &[]);
        assert_eq!(t.encode("the", SpecialMode::AsText), [257]);
        // `that` has "th" and "ha" overlapping; rank 0 ("th") goes first.
        assert_eq!(t.encode("that", SpecialMode::AsText), [256, A, T]);
        assert_eq!(t.encode("hat", SpecialMode::AsText), [258, T]);
        assert_eq!(t.vocab_size(), 259);
        assert_eq!(t.token_bytes(257).unwrap(), b"the");
    }

    #[test]
    fn a_repeated_pair_merges_left_to_right_without_overlap() {
        let t = tok(&[(A, A)], &[]);
        assert_eq!(t.encode("aaa", SpecialMode::AsText), [256, A]);
        assert_eq!(t.encode("aaaa", SpecialMode::AsText), [256, 256]);
        assert_eq!(t.encode("aaaaa", SpecialMode::AsText), [256, 256, A]);
    }

    #[test]
    fn merges_never_cross_pre_tokenizer_boundaries() {
        // 256 = "e " would join the end of a word to a space, but " " starts the next
        // word under the GPT-2 rules, so the two are in different pieces.
        let t = tok(&[(E, b' ' as u32)], &[]);
        assert_eq!(
            t.encode("the end", SpecialMode::AsText),
            [T, H, E, b' ' as u32, E, b'n' as u32, b'd' as u32]
        );
    }

    #[test]
    fn special_tokens_come_last_and_are_recognized_only_on_request() {
        let t = tok(&[(T, H)], &["<|eot|>", "<|pad|>"]);
        assert_eq!(t.vocab_size(), 256 + 1 + 2);
        assert_eq!(t.special_id("<|eot|>"), Some(257));
        assert_eq!(t.special_id("<|pad|>"), Some(258));
        assert_eq!(t.special_id("nope"), None);
        let text = "a<|eot|>b";
        assert_eq!(t.encode(text, SpecialMode::Recognize), [97, 257, 98]);
        // As text, the angle brackets and letters are encoded like any others.
        let plain = t.encode(text, SpecialMode::AsText);
        assert!(!plain.contains(&257) && t.decode(&plain).unwrap() == text);
    }

    #[test]
    fn special_tokens_at_the_edges_adjacent_and_overlapping() {
        let t = tok(&[], &["<a>", "<a>b", "<|x|>"]);
        let id = |s: &str| t.special_id(s).unwrap();
        assert_eq!(t.encode("<|x|>", SpecialMode::Recognize), [id("<|x|>")]);
        assert_eq!(t.encode("<|x|><|x|>", SpecialMode::Recognize), [id("<|x|>"), id("<|x|>")]);
        // The longest special at a position wins.
        assert_eq!(t.encode("<a>b", SpecialMode::Recognize), [id("<a>b")]);
        assert_eq!(t.encode("<a>c", SpecialMode::Recognize), [id("<a>"), 99]);
        // A partial special is plain text.
        assert_eq!(t.encode("<|x|", SpecialMode::Recognize).len(), 4);
        assert_eq!(t.encode("", SpecialMode::Recognize), Vec::<u32>::new());
    }

    #[test]
    fn decoding_returns_the_exact_bytes_and_names_a_bad_id() {
        let t = tok(&[(T, H)], &["<|eot|>"]);
        let ids = t.encode("the <|eot|>x", SpecialMode::Recognize);
        assert_eq!(t.decode(&ids).unwrap(), "the <|eot|>x");
        let err = t.decode_bytes(&[97, 258]).unwrap_err().to_string();
        assert!(err.contains("token id") && err.contains("258"), "{err}");
        assert!(t.decode_bytes(&[]).unwrap().is_empty());
    }

    #[test]
    fn decoding_text_replaces_invalid_utf8_but_bytes_are_exact() {
        let t = Tokenizer::bytes_only();
        let raw = [b'a', 0xff, 0xfe, b'b'];
        let ids = t.encode_bytes(&raw, SpecialMode::AsText);
        assert_eq!(t.decode_bytes(&ids).unwrap(), raw);
        assert_eq!(t.decode(&ids).unwrap(), "a\u{fffd}\u{fffd}b");
    }

    #[test]
    fn construction_rejects_impossible_merges_and_bad_specials() {
        let config = PreTokenizerConfig::gpt2_like();
        let build = |merges: Vec<(u32, u32)>, specials: Vec<&str>| {
            Tokenizer::new(config, merges, specials.into_iter().map(String::from).collect())
        };
        assert!(build(vec![(T, 256)], vec![]).is_err(), "uses its own id");
        assert!(build(vec![(T, 300)], vec![]).is_err(), "uses a future id");
        assert!(build(vec![(T, H), (T, H)], vec![]).is_err(), "repeats a pair");
        assert!(build(vec![(T, H), (256, 257)], vec![]).is_err());
        assert!(build(vec![], vec![""]).is_err());
        assert!(build(vec![], vec!["a\nb"]).is_err());
        assert!(build(vec![], vec!["a\rb"]).is_err());
        assert!(build(vec![], vec![" a"]).is_err());
        assert!(build(vec![], vec!["a "]).is_err());
        assert!(build(vec![], vec!["<x>", "<x>"]).is_err());
        assert!(build(vec![(T, H), (256, E)], vec!["<x>"]).is_ok());
    }

    fn sample_with_everything() -> Tokenizer {
        let config = PreTokenizerConfig {
            ansi: true,
            underscore_in_words: true,
            word_prefix: WordPrefix::AnyOne,
            digit_run: 3,
            punct_newlines: true,
            whitespace: WhitespaceRule::Cl100k,
        };
        Tokenizer::new(
            config,
            vec![(T, H), (256, E), (E, T)],
            vec!["<|endoftext|>".to_string(), "<| with spaces |>".to_string()],
        )
        .unwrap()
    }

    #[test]
    fn the_saved_text_format_has_a_fixed_layout_and_round_trips() {
        let t = sample_with_everything();
        let text = t.to_text();
        assert_eq!(
            text,
            "midman-tokenizer 1\n\
             pretokenizer ansi=1 underscore=1 prefix=any digits=3 punct_newlines=1 whitespace=cl100k\n\
             special <|endoftext|>\n\
             special <| with spaces |>\n\
             merge 116 104\n\
             merge 256 101\n\
             merge 101 116\n"
        );
        let back = Tokenizer::from_text(&text).unwrap();
        assert_eq!(back.merges(), t.merges());
        assert_eq!(back.special_tokens(), t.special_tokens());
        assert_eq!(back.pretokenizer(), t.pretokenizer());
        assert_eq!(back.to_text(), text);
        let sample = "the <|endoftext|> bet\x1b[0m";
        assert_eq!(
            back.encode(sample, SpecialMode::Recognize),
            t.encode(sample, SpecialMode::Recognize)
        );
        let empty = Tokenizer::bytes_only();
        assert_eq!(Tokenizer::from_text(&empty.to_text()).unwrap().vocab_size(), 256);
    }

    #[test]
    fn malformed_files_are_rejected_with_the_line_number() {
        let good = sample_with_everything().to_text();
        let lines: Vec<&str> = good.lines().collect();
        let with = |index: usize, replacement: &str| {
            let mut copy: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
            copy[index] = replacement.to_string();
            copy.join("\n") + "\n"
        };
        let bad: Vec<(&str, String)> = vec![
            ("empty", String::new()),
            ("wrong magic", with(0, "other-tokenizer 1")),
            ("no version", with(0, "midman-tokenizer")),
            ("version text", with(0, "midman-tokenizer one")),
            ("missing pretokenizer line", "midman-tokenizer 1\n".to_string()),
            ("wrong second line", with(1, "settings ansi=1")),
            ("setting without a value", with(1, "pretokenizer ansi")),
            ("unknown setting", with(1, "pretokenizer ansi=1 colour=1")),
            ("bad flag", with(1, &lines[1].replace("ansi=1", "ansi=2"))),
            ("bad prefix", with(1, &lines[1].replace("prefix=any", "prefix=many"))),
            ("bad digits", with(1, &lines[1].replace("digits=3", "digits=300"))),
            ("bad whitespace", with(1, &lines[1].replace("cl100k", "other"))),
            ("repeated setting", with(1, &format!("{} ansi=0", lines[1]))),
            ("missing setting", with(1, &lines[1].replace(" digits=3", ""))),
            ("merge with one number", with(4, "merge 116")),
            ("merge with three numbers", with(4, "merge 116 104 5")),
            ("merge with text", with(4, "merge a b")),
            ("merge with a negative number", with(4, "merge -1 104")),
            ("impossible merge", with(4, "merge 116 999")),
            ("blank line", with(4, "")),
            ("unknown line", with(4, "token 5")),
            ("special after merges", format!("{good}special <|late|>\n")),
        ];
        for (what, text) in bad {
            assert!(Tokenizer::from_text(&text).is_err(), "should reject: {what}");
        }
        assert!(Tokenizer::from_text(&good).is_ok());
        match Tokenizer::from_text(&with(1, "pretokenizer ansi=1 colour=1")) {
            Err(Error::Parse { line: 2, .. }) => {}
            other => panic!("expected a parse error on line 2, got {other:?}"),
        }
        match Tokenizer::from_text(&with(5, "merge 256")) {
            Err(Error::Parse { line: 6, .. }) => {}
            other => panic!("expected a parse error on line 6, got {other:?}"),
        }
    }

    #[test]
    fn a_future_version_is_an_unsupported_version_error() {
        let text = sample_with_everything().to_text().replacen(
            "midman-tokenizer 1",
            "midman-tokenizer 2",
            1,
        );
        match Tokenizer::from_text(&text) {
            Err(Error::UnsupportedVersion { found: 2, supported: 1, .. }) => {}
            other => panic!("expected an unsupported version, got {other:?}"),
        }
    }

    #[test]
    fn save_and_load_round_trip_through_a_file() {
        let path =
            std::env::temp_dir().join(format!("midman-tokenizer-{}.txt", std::process::id()));
        let t = sample_with_everything();
        t.save(&path).unwrap();
        let back = Tokenizer::load(&path).unwrap();
        assert_eq!(back.to_text(), t.to_text());
        std::fs::remove_file(&path).unwrap();
        let message = Tokenizer::load(&path).unwrap_err().to_string();
        assert!(message.contains("midman-tokenizer-"), "{message}");
    }

    /// The encoder written the slow obvious way: for each merge in rank order, replace
    /// every non-overlapping occurrence, left to right, in every piece.
    fn reference_encode(t: &Tokenizer, text: &[u8]) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in t.pretokenizer().pieces(text) {
            let mut ids: Vec<u32> = piece.iter().map(|&b| u32::from(b)).collect();
            for (rank, &(left, right)) in t.merges().iter().enumerate() {
                let new_id = FIRST_MERGE_ID + rank as u32;
                let mut next = Vec::with_capacity(ids.len());
                let mut i = 0;
                while i < ids.len() {
                    if i + 1 < ids.len() && ids[i] == left && ids[i + 1] == right {
                        next.push(new_id);
                        i += 2;
                    } else {
                        next.push(ids[i]);
                        i += 1;
                    }
                }
                ids = next;
            }
            out.extend(ids);
        }
        out
    }

    fn random_tokenizer(rng: &mut Rng, merges: usize) -> Tokenizer {
        let mut list: Vec<(u32, u32)> = Vec::new();
        let alphabet: Vec<u32> = b"abcde fgh_\n\t(){}01".iter().map(|&b| u32::from(b)).collect();
        while list.len() < merges {
            let limit = 256 + list.len();
            // Prefer small letters at first and recent tokens later, so merges nest.
            let pick = |rng: &mut Rng| -> u32 {
                if rng.below(3) == 0 && !list.is_empty() {
                    (256 + rng.below(list.len())) as u32
                } else if rng.below(4) == 0 {
                    rng.below(limit) as u32
                } else {
                    alphabet[rng.below(alphabet.len())]
                }
            };
            let pair = (pick(rng), pick(rng));
            if !list.contains(&pair) {
                list.push(pair);
            }
        }
        Tokenizer::new(PreTokenizerConfig::cl100k_like(), list, vec!["<|x|>".to_string()]).unwrap()
    }

    #[test]
    fn the_encoder_matches_the_obvious_reference_and_round_trips_on_random_data() {
        let mut rng = Rng::seed_from_u64(7);
        for case in 0..40 {
            let t = random_tokenizer(&mut rng, 5 + case * 4);
            for _ in 0..25 {
                let len = rng.below(60);
                let text: Vec<u8> = (0..len)
                    .map(|_| match rng.below(10) {
                        0 => 0x80 + rng.below(128) as u8,
                        1 => b'\n',
                        2 => b' ',
                        _ => b"abcdefgh_ (){}01\t"[rng.below(17)],
                    })
                    .collect();
                let ids = t.encode_bytes(&text, SpecialMode::AsText);
                assert_eq!(ids, reference_encode(&t, &text), "case {case}: {text:?}");
                assert_eq!(t.decode_bytes(&ids).unwrap(), text, "case {case}: {text:?}");
            }
        }
    }
}
