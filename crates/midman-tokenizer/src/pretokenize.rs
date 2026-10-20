// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tokenizer.md, section "pretokenize.rs"
// ============================================================================
//! The pre-tokenizer: splits text into pieces that byte-pair merges never cross.
//!
//! Without it a merge could join the end of one word to the start of the next, and the
//! vocabulary would fill with fragments like `e_t` or `);\n`. The rules here are a
//! hand-written scanner for a small family of GPT-2 style patterns, chosen by a
//! [`PreTokenizerConfig`] so that variants can be trained and compared on real data.
//!
//! The rules are tried in this order at each position, and the first that matches
//! decides the piece:
//!
//! 1. an ANSI escape sequence (`ESC [ ... final byte`) or a lone `ESC`, when `ansi` is on;
//! 2. a word: letters (and `_` when `underscore_in_words` is on), with an optional
//!    prefix, which is one space or, with [`WordPrefix::AnyOne`], any one character
//!    that is not a letter, digit or line break;
//! 3. digits, at most `digit_run` at a time (`0` means no limit), with an optional
//!    space prefix under [`WordPrefix::Space`];
//! 4. a run of punctuation and symbols, with an optional space prefix and, when
//!    `punct_newlines` is on, the line breaks that follow;
//! 5. whitespace, by one of two rules ([`WhitespaceRule`]).
//!
//! A letter is a Unicode alphabetic character, a digit is an ASCII digit, and
//! whitespace is the Unicode `White_Space` property. Bytes that are not valid UTF-8
//! count as punctuation, one byte at a time, so any byte string can be split and the
//! pieces always concatenate back to the input.

/// What may come before a word, and before a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordPrefix {
    /// An optional single space, as in GPT-2. Numbers also take an optional space.
    Space,
    /// Any one character that is not a letter, digit or line break, as in
    /// `cl100k_base`. Paths like `/usr`, flags like `-v` and fields like `.name` become
    /// one piece each. Numbers take no prefix.
    AnyOne,
}

/// How runs of whitespace are cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhitespaceRule {
    /// A run of whitespace is one piece, except that its last character is left to
    /// become the prefix of the next word. A line break and the indentation after it
    /// therefore form one piece.
    Gpt2,
    /// As `Gpt2`, but a run that contains a line break ends at its last line break, so
    /// blank lines and trailing spaces form one piece and the next line's indentation
    /// starts a new one.
    Cl100k,
}

/// The pre-tokenizer's settings. They are stored in a saved tokenizer, because encoding
/// must split text the same way the vocabulary was trained on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreTokenizerConfig {
    /// Treat `ESC [ ... final byte` as one piece, so terminal colors do not scatter.
    pub ansi: bool,
    /// Count `_` as a letter, so `snake_case` is one word instead of three pieces.
    pub underscore_in_words: bool,
    /// What may prefix a word.
    pub word_prefix: WordPrefix,
    /// The most digits in one piece, or `0` for no limit.
    pub digit_run: u8,
    /// Let a run of punctuation absorb the line breaks right after it, so `{\n` and
    /// `;\n` are single pieces.
    pub punct_newlines: bool,
    /// How whitespace runs are cut.
    pub whitespace: WhitespaceRule,
}

impl PreTokenizerConfig {
    /// The GPT-2 rules without contractions.
    pub fn gpt2_like() -> PreTokenizerConfig {
        PreTokenizerConfig {
            ansi: false,
            underscore_in_words: false,
            word_prefix: WordPrefix::Space,
            digit_run: 0,
            punct_newlines: false,
            whitespace: WhitespaceRule::Gpt2,
        }
    }

    /// The `cl100k_base` rules without contractions.
    pub fn cl100k_like() -> PreTokenizerConfig {
        PreTokenizerConfig {
            ansi: false,
            underscore_in_words: false,
            word_prefix: WordPrefix::AnyOne,
            digit_run: 3,
            punct_newlines: true,
            whitespace: WhitespaceRule::Cl100k,
        }
    }

    /// The settings chosen for code and terminal text: ANSI escape sequences are whole
    /// pieces, identifiers with underscores stay whole, words and paths take any one
    /// character before them, numbers are cut every three digits, punctuation absorbs
    /// the line breaks after it, and a line break is joined with the indentation after
    /// it. On a mix of Rust, shell, terminal sessions, diffs, configuration, prose and
    /// Ubel text, this scored best of the eight settings compared; see
    /// `docs/midman-tokenizer.md`.
    pub fn code_shell() -> PreTokenizerConfig {
        PreTokenizerConfig {
            ansi: true,
            underscore_in_words: true,
            word_prefix: WordPrefix::AnyOne,
            digit_run: 3,
            punct_newlines: true,
            whitespace: WhitespaceRule::Gpt2,
        }
    }

    /// Splits `text` into pieces. The pieces are slices of `text` that concatenate back
    /// to it exactly, none is empty and none is longer than [`MAX_PIECE_BYTES`].
    pub fn pieces<'a>(&self, text: &'a [u8]) -> Pieces<'a> {
        Pieces { config: *self, text, pos: 0, token_end: 0 }
    }
}

impl Default for PreTokenizerConfig {
    /// The settings chosen for code and terminal text: [`PreTokenizerConfig::code_shell`].
    fn default() -> PreTokenizerConfig {
        PreTokenizerConfig::code_shell()
    }
}

/// No piece is longer than this many bytes. A longer match (a base64 blob, a minified
/// line, a wall of `=`) is cut into pieces of this size, at a character boundary when
/// one is near. It bounds the cost of training and encoding to a fixed amount per
/// piece, and affects nothing but such runs.
pub const MAX_PIECE_BYTES: usize = 128;

/// The iterator returned by [`PreTokenizerConfig::pieces`].
#[derive(Clone, Debug)]
pub struct Pieces<'a> {
    config: PreTokenizerConfig,
    text: &'a [u8],
    pos: usize,
    /// The end of the match being handed out in chunks, when it was over the limit.
    token_end: usize,
}

impl<'a> Iterator for Pieces<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        if self.pos >= self.text.len() {
            return None;
        }
        if self.pos >= self.token_end {
            self.token_end = self.pos + self.config.match_len(self.text, self.pos);
        }
        let mut end = self.token_end;
        if end - self.pos > MAX_PIECE_BYTES {
            end = self.pos + MAX_PIECE_BYTES;
            // Move the cut back by up to three bytes so it does not fall inside a
            // multi-byte character. In bytes that are not UTF-8 it stays where it is.
            let mut cut = end;
            while end - cut < 3 && is_continuation(self.text[cut]) {
                cut -= 1;
            }
            if !is_continuation(self.text[cut]) {
                end = cut;
            }
        }
        let piece = &self.text[self.pos..end];
        self.pos = end;
        Some(piece)
    }
}

fn is_continuation(byte: u8) -> bool {
    byte & 0xc0 == 0x80
}

/// What one character is, as far as the rules care.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Word,
    Digit,
    Punct,
    /// U+0020 only: the one whitespace character that can prefix a word.
    Space,
    /// `\n` or `\r`.
    Newline,
    /// Any other whitespace: tab, vertical tab, form feed, no-break space and so on.
    OtherWhitespace,
    /// `ESC`, when ANSI handling is on.
    Esc,
}

impl Kind {
    fn is_whitespace(self) -> bool {
        matches!(self, Kind::Space | Kind::Newline | Kind::OtherWhitespace)
    }
}

impl PreTokenizerConfig {
    /// The kind and byte length of the character at `i`, which must be in range.
    fn classify(&self, text: &[u8], i: usize) -> (Kind, usize) {
        let byte = text[i];
        if byte < 0x80 {
            let kind = match byte {
                b'a'..=b'z' | b'A'..=b'Z' => Kind::Word,
                b'0'..=b'9' => Kind::Digit,
                b' ' => Kind::Space,
                b'\n' | b'\r' => Kind::Newline,
                0x09 | 0x0b | 0x0c => Kind::OtherWhitespace,
                b'_' if self.underscore_in_words => Kind::Word,
                0x1b if self.ansi => Kind::Esc,
                _ => Kind::Punct,
            };
            return (kind, 1);
        }
        let want = match byte {
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            // A continuation byte with no lead, or a byte UTF-8 never uses.
            _ => return (Kind::Punct, 1),
        };
        let Some(bytes) = text.get(i..i + want) else {
            return (Kind::Punct, 1);
        };
        match std::str::from_utf8(bytes).ok().and_then(|s| s.chars().next()) {
            Some(c) if c.is_alphabetic() => (Kind::Word, want),
            Some(c) if c.is_whitespace() => (Kind::OtherWhitespace, want),
            Some(_) => (Kind::Punct, want),
            None => (Kind::Punct, 1),
        }
    }

    /// The length of the match at `i`, which is at least one byte.
    fn match_len(&self, text: &[u8], i: usize) -> usize {
        let (kind, len) = self.classify(text, i);
        if kind == Kind::Esc {
            return ansi_len(text, i).unwrap_or(1);
        }
        self.word_len(text, i, kind, len)
            .or_else(|| self.digits_len(text, i, kind, len))
            .or_else(|| self.punct_len(text, i, kind, len))
            .unwrap_or_else(|| self.whitespace_len(text, i))
            .max(1)
    }

    /// Consumes characters from `j` while they are of `kind`, up to `limit` of them.
    fn run(&self, text: &[u8], mut j: usize, kind: Kind, limit: usize) -> usize {
        let mut taken = 0;
        while j < text.len() && taken < limit {
            let (k, l) = self.classify(text, j);
            if k != kind {
                break;
            }
            j += l;
            taken += 1;
        }
        j
    }

    fn word_len(&self, text: &[u8], i: usize, kind: Kind, len: usize) -> Option<usize> {
        let prefixed = match self.word_prefix {
            WordPrefix::Space => kind == Kind::Space,
            WordPrefix::AnyOne => !matches!(kind, Kind::Word | Kind::Digit | Kind::Newline),
        };
        let start = if prefixed {
            let next = i + len;
            if next < text.len() && self.classify(text, next).0 == Kind::Word {
                next
            } else {
                return None;
            }
        } else if kind == Kind::Word {
            i
        } else {
            return None;
        };
        Some(self.run(text, start, Kind::Word, usize::MAX) - i)
    }

    fn digits_len(&self, text: &[u8], i: usize, kind: Kind, len: usize) -> Option<usize> {
        let start = if kind == Kind::Digit {
            i
        } else if kind == Kind::Space && self.word_prefix == WordPrefix::Space {
            let next = i + len;
            if next < text.len() && self.classify(text, next).0 == Kind::Digit {
                next
            } else {
                return None;
            }
        } else {
            return None;
        };
        let limit = if self.digit_run == 0 { usize::MAX } else { usize::from(self.digit_run) };
        Some(self.run(text, start, Kind::Digit, limit) - i)
    }

    fn punct_len(&self, text: &[u8], i: usize, kind: Kind, len: usize) -> Option<usize> {
        let start = if kind == Kind::Punct {
            i
        } else if kind == Kind::Space {
            let next = i + len;
            if next < text.len() && self.classify(text, next).0 == Kind::Punct {
                next
            } else {
                return None;
            }
        } else {
            return None;
        };
        let mut end = self.run(text, start, Kind::Punct, usize::MAX);
        if self.punct_newlines {
            end = self.run(text, end, Kind::Newline, usize::MAX);
        }
        Some(end - i)
    }

    /// The whitespace rules. Called only at a whitespace character, which no other rule
    /// claimed.
    fn whitespace_len(&self, text: &[u8], i: usize) -> usize {
        let mut j = i;
        let mut last_start = i;
        let mut count = 0usize;
        let mut after_last_newline = None;
        while j < text.len() {
            let (kind, len) = self.classify(text, j);
            if !kind.is_whitespace() {
                break;
            }
            last_start = j;
            count += 1;
            j += len;
            if kind == Kind::Newline {
                after_last_newline = Some(j);
            }
        }
        if self.whitespace == WhitespaceRule::Cl100k {
            if let Some(end) = after_last_newline {
                return end - i;
            }
        }
        if j == text.len() {
            // The whole run, since nothing follows it to take the last character.
            j - i
        } else if count >= 2 {
            // Leave the last whitespace character for the next piece.
            last_start - i
        } else {
            j - i
        }
    }
}

/// The length of the escape sequence at `i`: `ESC [`, parameter bytes `0-?`,
/// intermediate bytes ` -/`, and a final byte `@-~`. `None` if it is not complete.
fn ansi_len(text: &[u8], i: usize) -> Option<usize> {
    if text.get(i + 1) != Some(&b'[') {
        return None;
    }
    let mut j = i + 2;
    while j < text.len() && (0x30..=0x3f).contains(&text[j]) {
        j += 1;
    }
    while j < text.len() && (0x20..=0x2f).contains(&text[j]) {
        j += 1;
    }
    if j < text.len() && (0x40..=0x7e).contains(&text[j]) {
        Some(j + 1 - i)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(config: &PreTokenizerConfig, text: &str) -> Vec<String> {
        config.pieces(text.as_bytes()).map(|p| String::from_utf8_lossy(p).into_owned()).collect()
    }

    fn gpt2(text: &str) -> Vec<String> {
        split(&PreTokenizerConfig::gpt2_like(), text)
    }

    fn cl100k(text: &str) -> Vec<String> {
        split(&PreTokenizerConfig::cl100k_like(), text)
    }

    fn all_configs() -> Vec<PreTokenizerConfig> {
        let mut configs = Vec::new();
        for ansi in [false, true] {
            for underscore_in_words in [false, true] {
                for word_prefix in [WordPrefix::Space, WordPrefix::AnyOne] {
                    for digit_run in [0u8, 1, 3] {
                        for punct_newlines in [false, true] {
                            for whitespace in [WhitespaceRule::Gpt2, WhitespaceRule::Cl100k] {
                                configs.push(PreTokenizerConfig {
                                    ansi,
                                    underscore_in_words,
                                    word_prefix,
                                    digit_run,
                                    punct_newlines,
                                    whitespace,
                                });
                            }
                        }
                    }
                }
            }
        }
        configs
    }

    #[test]
    fn gpt2_rules_split_words_numbers_and_punctuation() {
        assert_eq!(gpt2("Hello world"), ["Hello", " world"]);
        assert_eq!(gpt2("x = 12;"), ["x", " =", " 12", ";"]);
        assert_eq!(gpt2("a->b::c"), ["a", "->", "b", "::", "c"]);
        assert_eq!(gpt2("snake_case"), ["snake", "_", "case"]);
        assert_eq!(gpt2("a  b"), ["a", " ", " b"]);
        assert_eq!(gpt2("1234567"), ["1234567"]);
    }

    #[test]
    fn gpt2_whitespace_joins_a_line_break_with_the_indent_but_one_space() {
        assert_eq!(
            gpt2("fn main() {\n    let x = 1;\n}\n"),
            ["fn", " main", "()", " {", "\n   ", " let", " x", " =", " 1", ";", "\n", "}", "\n"]
        );
        // A run at the very end is taken whole.
        assert_eq!(gpt2("a  \n  "), ["a", "  \n  "]);
    }

    #[test]
    fn cl100k_rules_prefix_words_with_any_one_character_and_cap_digits() {
        assert_eq!(cl100k("/usr/local/bin"), ["/usr", "/local", "/bin"]);
        assert_eq!(cl100k("--release -v"), ["--", "release", " -", "v"]);
        assert_eq!(cl100k("a.b(c)"), ["a", ".b", "(c", ")"]);
        assert_eq!(cl100k("x = 1234567"), ["x", " =", " ", "123", "456", "7"]);
    }

    #[test]
    fn cl100k_whitespace_ends_at_the_last_line_break() {
        assert_eq!(cl100k("a  \n\n    b"), ["a", "  \n\n", "   ", " b"]);
        assert_eq!(cl100k("{\n    x"), ["{\n", "   ", " x"]);
        assert_eq!(cl100k("x;\n"), ["x", ";\n"]);
    }

    #[test]
    fn ansi_sequences_are_single_pieces_when_enabled() {
        let on = PreTokenizerConfig { ansi: true, ..PreTokenizerConfig::cl100k_like() };
        let text = "\x1b[1;32mok\x1b[0m done)\x1b[K";
        assert_eq!(split(&on, text), ["\x1b[1;32m", "ok", "\x1b[0m", " done", ")", "\x1b[K"]);
        // Off, the escape is punctuation like any other and glues to its neighbors.
        let off = PreTokenizerConfig::cl100k_like();
        assert_eq!(split(&off, "\x1b[0m"), ["\x1b[", "0", "m"]);
    }

    #[test]
    fn an_incomplete_escape_sequence_is_a_lone_escape() {
        let on = PreTokenizerConfig { ansi: true, ..PreTokenizerConfig::cl100k_like() };
        assert_eq!(split(&on, "a\x1b"), ["a", "\x1b"]);
        assert_eq!(split(&on, "\x1b[12"), ["\x1b", "[", "12"]);
        // `;t` is one piece: under the any-one-character prefix rule a word may start with
        // the punctuation before it.
        assert_eq!(split(&on, "\x1b]0;t"), ["\x1b", "]", "0", ";t"]);
    }

    #[test]
    fn the_underscore_setting_decides_whether_identifiers_stay_whole() {
        let on =
            PreTokenizerConfig { underscore_in_words: true, ..PreTokenizerConfig::gpt2_like() };
        assert_eq!(
            split(&on, "let snake_case_name = _x;"),
            ["let", " snake_case_name", " =", " _x", ";"]
        );
        assert_eq!(split(&on, "__init__"), ["__init__"]);
    }

    #[test]
    fn digit_runs_are_cut_at_the_configured_length() {
        let one = PreTokenizerConfig { digit_run: 1, ..PreTokenizerConfig::gpt2_like() };
        assert_eq!(split(&one, "ab 123"), ["ab", " 1", "2", "3"]);
        let unlimited = PreTokenizerConfig::gpt2_like();
        assert_eq!(split(&unlimited, "ab 123"), ["ab", " 123"]);
    }

    #[test]
    fn vertical_tab_and_form_feed_are_whitespace_not_punctuation() {
        assert_eq!(gpt2("a\x0b;"), ["a", "\x0b", ";"]);
        assert_eq!(gpt2("a\x0c;"), ["a", "\x0c", ";"]);
        // The other control characters are punctuation, so they join a run.
        assert_eq!(gpt2("a\x01;"), ["a", "\x01;"]);
    }

    #[test]
    fn non_ascii_text_follows_unicode_classes() {
        assert_eq!(gpt2("héllo wörld"), ["héllo", " wörld"]);
        assert_eq!(gpt2("日本語 text"), ["日本語", " text"]);
        // A no-break space is whitespace but not the one that prefixes a word.
        assert_eq!(gpt2("a\u{a0}b"), ["a", "\u{a0}", "b"]);
        // An arrow and an emoji are symbols, grouped as punctuation.
        assert_eq!(gpt2("a→b"), ["a", "→", "b"]);
        assert_eq!(gpt2("ok 👍👍"), ["ok", " 👍👍"]);
    }

    #[test]
    fn bytes_that_are_not_utf8_become_punctuation_one_at_a_time() {
        let config = PreTokenizerConfig::gpt2_like();
        let text = [b'f', 0xff, 0xfe, b'o', 0xe2, 0x82, b'x'];
        let pieces: Vec<&[u8]> = config.pieces(&text).collect();
        let want: Vec<&[u8]> = vec![b"f", &[0xff, 0xfe], b"o", &[0xe2, 0x82], b"x"];
        assert_eq!(pieces, want);
    }

    #[test]
    fn pieces_always_concatenate_to_the_input_for_every_setting() {
        let samples: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b" ".to_vec(),
            b"\n".to_vec(),
            b"\x1b".to_vec(),
            b"  \t \r\n \n  x_y 12 ;;\n\x1b[0m\x1b[31;1mred\x1b[0m  ".to_vec(),
            "naïve café — 日本 \u{a0}\u{3000}x\u{301}".as_bytes().to_vec(),
            vec![0x80, 0xff, 0xc3, 0x28, 0xe2, 0x82, 0xf0, 0x9f, 0x98],
            (0u8..=255).collect(),
        ];
        for config in all_configs() {
            for sample in &samples {
                let joined: Vec<u8> = config.pieces(sample).flatten().copied().collect();
                assert_eq!(&joined, sample, "{config:?}");
                assert!(config.pieces(sample).all(|p| !p.is_empty() && p.len() <= MAX_PIECE_BYTES));
            }
        }
    }

    #[test]
    fn long_matches_are_cut_at_the_limit_and_at_character_boundaries() {
        let config = PreTokenizerConfig::gpt2_like();
        let long = "a".repeat(300);
        let sizes: Vec<usize> = config.pieces(long.as_bytes()).map(<[u8]>::len).collect();
        assert_eq!(sizes, [128, 128, 44]);

        // 100 ASCII letters then two-byte letters: the cut at 128 would fall between the
        // bytes of a character, so it moves back to the character's start.
        let mixed = format!("{}{}", "a".repeat(127), "é".repeat(40));
        let pieces: Vec<&[u8]> = config.pieces(mixed.as_bytes()).collect();
        assert!(pieces.iter().all(|p| std::str::from_utf8(p).is_ok()), "{pieces:?}");
        assert_eq!(pieces[0].len(), 127);
        let joined: Vec<u8> = pieces.concat();
        assert_eq!(joined, mixed.as_bytes());

        // Bytes that are not UTF-8 are cut exactly at the limit.
        let raw = vec![0xffu8; 300];
        let sizes: Vec<usize> = config.pieces(&raw).map(<[u8]>::len).collect();
        assert_eq!(sizes, [128, 128, 44]);
    }

    #[test]
    fn the_default_is_the_code_and_shell_setting_the_measurements_chose() {
        let c = PreTokenizerConfig::default();
        assert_eq!(c, PreTokenizerConfig::code_shell());
        assert!(c.ansi && c.underscore_in_words && c.punct_newlines);
        assert_eq!(
            (c.word_prefix, c.digit_run, c.whitespace),
            (WordPrefix::AnyOne, 3, WhitespaceRule::Gpt2)
        );
        assert_eq!(
            split(&c, "$ cargo test --release\n\x1b[32mok\x1b[0m snake_case 12345\n    x"),
            [
                "$",
                " cargo",
                " test",
                " --",
                "release",
                "\n",
                "\x1b[32m",
                "ok",
                "\x1b[0m",
                " snake_case",
                " ",
                "123",
                "45",
                "\n   ",
                " x"
            ]
        );
    }

    #[test]
    fn a_cut_inside_a_four_byte_character_moves_back_up_to_three_bytes() {
        let config = PreTokenizerConfig::gpt2_like();
        // U+20000 is a letter four bytes long. The first one starts at byte 125, so a cut
        // at 128 lands on its last continuation byte and has to move back three bytes.
        let text = format!("{}{}", "a".repeat(125), "\u{20000}".repeat(40));
        let pieces: Vec<&[u8]> = config.pieces(text.as_bytes()).collect();
        assert_eq!(pieces[0].len(), 125);
        assert!(pieces.iter().all(|p| std::str::from_utf8(p).is_ok()));
        assert_eq!(pieces.concat(), text.as_bytes());
    }

    #[test]
    fn splitting_is_deterministic_and_only_depends_on_the_text() {
        let config = PreTokenizerConfig::cl100k_like();
        let text = b"let x = foo(bar, 12);\n";
        let a: Vec<&[u8]> = config.pieces(text).collect();
        let b: Vec<&[u8]> = config.pieces(text).collect();
        assert_eq!(a, b);
    }
}
