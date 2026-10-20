// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-tokenizer.md, section "trainer.rs"
// ============================================================================
//! The byte-pair-encoding trainer.
//!
//! Text is fed in with [`Trainer::add`], which cuts it into pieces with the
//! pre-tokenizer and counts how often each distinct piece occurs. [`Trainer::finish`]
//! then learns merges from those counts, never from the text again, so memory depends
//! on the number of distinct pieces and not on the size of the corpus.
//!
//! Each step merges the most frequent adjacent pair of tokens. Ties go to the pair with
//! the smaller left id and then the smaller right id, so the result does not depend on
//! the order text was added or on hash order. Pair counts count every adjacent
//! position, so in `aaa` the pair `aa` counts twice although only one merge can apply;
//! the counts are exact in every other way, and the reference algorithm counts the same.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use midman_foundation::{Error, Result};

use crate::pretokenize::PreTokenizerConfig;
use crate::tokenizer::{validate_specials, Tokenizer, FIRST_MERGE_ID};

/// What to train.
#[derive(Clone, Debug)]
pub struct TrainerConfig {
    /// The total vocabulary: 256 bytes, the merges, and the special tokens.
    pub vocab_size: usize,
    /// How text is split into pieces, which fixes what merges can exist.
    pub pretokenizer: PreTokenizerConfig,
    /// Special tokens, given the last ids.
    pub special_tokens: Vec<String>,
    /// A pair seen fewer times than this is never merged, so the vocabulary may end up
    /// smaller than requested on a small corpus.
    pub min_frequency: u64,
}

impl TrainerConfig {
    /// A configuration for `vocab_size` tokens with the default pre-tokenizer, no
    /// special tokens and a minimum frequency of 2.
    pub fn new(vocab_size: usize) -> TrainerConfig {
        TrainerConfig {
            vocab_size,
            pretokenizer: PreTokenizerConfig::default(),
            special_tokens: Vec::new(),
            min_frequency: 2,
        }
    }
}

/// What a training run did.
#[derive(Clone, Debug)]
pub struct TrainStats {
    /// The merges learned.
    pub merges: usize,
    /// The merges that were asked for: the vocabulary minus bytes and special tokens.
    pub requested_merges: usize,
    /// The pair count of each merge when it was made, in order. It never increases.
    pub merge_counts: Vec<u64>,
    /// How many distinct pieces the corpus had.
    pub unique_pieces: usize,
    /// How many pieces the corpus had in all.
    pub pieces_seen: u64,
    /// How many bytes the corpus had in all.
    pub bytes_seen: u64,
}

impl TrainStats {
    /// True if training ran out of pairs that meet the minimum frequency before it
    /// reached the requested vocabulary.
    pub fn stopped_early(&self) -> bool {
        self.merges < self.requested_merges
    }
}

/// Counts pieces of a corpus and learns merges from the counts.
#[derive(Debug)]
pub struct Trainer {
    config: TrainerConfig,
    counts: HashMap<Box<[u8]>, u64>,
    pieces_seen: u64,
    bytes_seen: u64,
}

impl Trainer {
    /// Starts a training run.
    ///
    /// # Errors
    ///
    /// Returns an error if the vocabulary has no room for the 256 bytes and the special
    /// tokens, if the minimum frequency is zero, or if a special token is not valid
    /// (see [`Tokenizer::new`]).
    pub fn new(config: TrainerConfig) -> Result<Trainer> {
        validate_specials(&config.special_tokens)?;
        let fixed = 256 + config.special_tokens.len();
        if config.vocab_size < fixed {
            return Err(Error::invalid(format!(
                "a vocabulary of {} cannot hold the 256 bytes and {} special tokens",
                config.vocab_size,
                config.special_tokens.len()
            )));
        }
        if config.min_frequency == 0 {
            return Err(Error::invalid("the minimum frequency must be at least 1"));
        }
        Ok(Trainer { config, counts: HashMap::new(), pieces_seen: 0, bytes_seen: 0 })
    }

    /// Adds a document or a chunk of one. Pieces never span two calls, so pass whole
    /// documents or whole lines.
    pub fn add(&mut self, text: &[u8]) {
        let config = self.config.pretokenizer;
        for piece in config.pieces(text) {
            self.add_counted(piece, 1);
        }
    }

    /// Adds a piece that occurred `count` times, for callers that already counted.
    pub fn add_counted(&mut self, piece: &[u8], count: u64) {
        if piece.is_empty() || count == 0 {
            return;
        }
        self.pieces_seen += count;
        self.bytes_seen += piece.len() as u64 * count;
        match self.counts.get_mut(piece) {
            Some(existing) => *existing += count,
            None => {
                self.counts.insert(piece.into(), count);
            }
        }
    }

    /// Learns the merges and builds the tokenizer.
    ///
    /// # Errors
    ///
    /// Returns an error only if the tokenizer cannot be built, which would be a bug.
    pub fn finish(self) -> Result<(Tokenizer, TrainStats)> {
        let requested = self.config.vocab_size - 256 - self.config.special_tokens.len();
        let unique_pieces = self.counts.len();
        let words: Vec<Word> = self
            .counts
            .into_iter()
            .map(|(bytes, count)| Word {
                symbols: bytes.iter().map(|&b| u32::from(b)).collect(),
                count: i64::try_from(count).unwrap_or(i64::MAX),
            })
            .collect();
        let learned = learn(words, requested, self.config.min_frequency);
        let stats = TrainStats {
            merges: learned.merges.len(),
            requested_merges: requested,
            merge_counts: learned.counts,
            unique_pieces,
            pieces_seen: self.pieces_seen,
            bytes_seen: self.bytes_seen,
        };
        let tokenizer =
            Tokenizer::new(self.config.pretokenizer, learned.merges, self.config.special_tokens)?;
        Ok((tokenizer, stats))
    }
}

/// One distinct piece, as the tokens it currently consists of.
#[derive(Clone, Debug)]
struct Word {
    symbols: Vec<u32>,
    count: i64,
}

/// A pair waiting in the priority queue, with the count it had when it was queued.
#[derive(PartialEq, Eq)]
struct Candidate {
    count: i64,
    pair: (u32, u32),
}

impl Ord for Candidate {
    /// The highest count first; among equal counts, the smaller pair first.
    fn cmp(&self, other: &Candidate) -> Ordering {
        self.count.cmp(&other.count).then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Candidate) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The result of learning: merges in rank order, their counts, and each word as it
/// ended up.
struct Learned {
    merges: Vec<(u32, u32)>,
    counts: Vec<u64>,
    /// Only the tests look at how each word ended up.
    #[cfg(test)]
    words: Vec<Word>,
}

fn contains_pair(symbols: &[u32], pair: (u32, u32)) -> bool {
    symbols.windows(2).any(|w| w[0] == pair.0 && w[1] == pair.1)
}

/// Replaces every non-overlapping occurrence of `pair`, left to right.
fn merge_in_place(symbols: &mut Vec<u32>, pair: (u32, u32), new_id: u32) {
    let (mut read, mut write) = (0, 0);
    while read < symbols.len() {
        if read + 1 < symbols.len() && symbols[read] == pair.0 && symbols[read + 1] == pair.1 {
            symbols[write] = new_id;
            read += 2;
        } else {
            symbols[write] = symbols[read];
            read += 1;
        }
        write += 1;
    }
    symbols.truncate(write);
}

/// Learns up to `target` merges from the words.
///
/// The index `pair_words` lists, for each pair, words that contain it. It may also list
/// words that no longer do, which is checked when the pair is merged, but it never
/// misses a word that does.
fn learn(mut words: Vec<Word>, target: usize, min_frequency: u64) -> Learned {
    let min_frequency = i64::try_from(min_frequency).unwrap_or(i64::MAX);
    let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::new();
    let mut pair_words: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (index, word) in words.iter().enumerate() {
        for w in word.symbols.windows(2) {
            let pair = (w[0], w[1]);
            *pair_counts.entry(pair).or_insert(0) += word.count;
            let list = pair_words.entry(pair).or_default();
            if list.last() != Some(&(index as u32)) {
                list.push(index as u32);
            }
        }
    }
    let mut heap: BinaryHeap<Candidate> =
        pair_counts.iter().map(|(&pair, &count)| Candidate { count, pair }).collect();

    let mut merges: Vec<(u32, u32)> = Vec::new();
    let mut merge_counts: Vec<u64> = Vec::new();
    while merges.len() < target {
        let Some(top) = heap.pop() else { break };
        let current = pair_counts.get(&top.pair).copied().unwrap_or(0);
        if current != top.count {
            // The count changed since this entry was queued; queue the true one.
            if current > 0 {
                heap.push(Candidate { count: current, pair: top.pair });
            }
            continue;
        }
        if current < min_frequency {
            break;
        }
        let pair = top.pair;
        let new_id = FIRST_MERGE_ID + merges.len() as u32;
        merges.push(pair);
        merge_counts.push(current as u64);

        let mut indices = pair_words.remove(&pair).unwrap_or_default();
        indices.sort_unstable();
        indices.dedup();
        let mut created: HashSet<(u32, u32)> = HashSet::new();
        for index in indices {
            let word = &mut words[index as usize];
            if !contains_pair(&word.symbols, pair) {
                continue;
            }
            for w in word.symbols.windows(2) {
                let key = (w[0], w[1]);
                if let Some(count) = pair_counts.get_mut(&key) {
                    *count -= word.count;
                    if *count == 0 {
                        pair_counts.remove(&key);
                    }
                }
            }
            merge_in_place(&mut word.symbols, pair, new_id);
            for w in word.symbols.windows(2) {
                let key = (w[0], w[1]);
                *pair_counts.entry(key).or_insert(0) += word.count;
                if key.0 == new_id || key.1 == new_id {
                    created.insert(key);
                    let list = pair_words.entry(key).or_default();
                    if list.last() != Some(&index) {
                        list.push(index);
                    }
                }
            }
        }
        for key in created {
            if let Some(&count) = pair_counts.get(&key) {
                heap.push(Candidate { count, pair: key });
            }
        }
    }
    Learned {
        merges,
        counts: merge_counts,
        #[cfg(test)]
        words,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::SpecialMode;
    use midman_foundation::Rng;

    /// The textbook example (Sennrich et al.): low x5, lower x2, newest x6, widest x3.
    /// The merges below were worked out by hand with this crate's tie rule.
    #[test]
    fn learns_the_textbook_example_in_the_hand_derived_order() {
        let mut trainer = Trainer::new(TrainerConfig::new(256 + 8)).unwrap();
        trainer.add_counted(b"low", 5);
        trainer.add_counted(b"lower", 2);
        trainer.add_counted(b"newest", 6);
        trainer.add_counted(b"widest", 3);
        let (tokenizer, stats) = trainer.finish().unwrap();
        let (e, s, t, l, o, w, n, d) = (101, 115, 116, 108, 111, 119, 110, 100);
        assert_eq!(
            tokenizer.merges(),
            [(e, s), (256, t), (l, o), (258, w), (e, w), (n, 260), (261, 257), (d, 257)],
            "es, est, lo, low, ew, new, newest, dest"
        );
        assert_eq!(stats.merge_counts, [9, 9, 7, 7, 6, 6, 6, 3]);
        assert!(!stats.stopped_early());
        assert_eq!(stats.unique_pieces, 4);
        assert_eq!(stats.pieces_seen, 16);
        assert_eq!(stats.bytes_seen, 5 * 3 + 2 * 5 + 6 * 6 + 3 * 6);
        assert_eq!(tokenizer.token_bytes(262).unwrap(), b"newest");
    }

    #[test]
    fn the_result_does_not_depend_on_the_order_text_was_added_or_on_hash_order() {
        let texts: Vec<String> = (0..30)
            .map(|i| format!("let value_{i} = compute(a, b);\n    if x > {i} {{ return; }}\n"))
            .collect();
        let train = |order: &[usize]| {
            let mut config = TrainerConfig::new(256 + 40);
            config.min_frequency = 1;
            let mut trainer = Trainer::new(config).unwrap();
            for &i in order {
                trainer.add(texts[i].as_bytes());
            }
            trainer.finish().unwrap().0.merges().to_vec()
        };
        let forward: Vec<usize> = (0..30).collect();
        let mut shuffled = forward.clone();
        Rng::seed_from_u64(3).shuffle(&mut shuffled);
        let a = train(&forward);
        assert_eq!(a, train(&shuffled));
        assert_eq!(a, train(&forward), "a second run with a new hash seed");
        assert_eq!(a.len(), 40);
    }

    #[test]
    fn a_corpus_with_no_frequent_pairs_stops_early_and_says_so() {
        let mut trainer = Trainer::new(TrainerConfig::new(1000)).unwrap();
        trainer.add(b"abcdefg");
        let (tokenizer, stats) = trainer.finish().unwrap();
        assert!(tokenizer.merges().is_empty());
        assert_eq!(tokenizer.vocab_size(), 256);
        assert!(stats.stopped_early());
        assert_eq!(stats.requested_merges, 744);
    }

    #[test]
    fn the_minimum_frequency_is_respected() {
        let corpus: &[&[u8]] = &[b"ab", b"ab", b"cd"];
        let run = |min_frequency: u64| {
            let mut config = TrainerConfig::new(300);
            config.min_frequency = min_frequency;
            let mut trainer = Trainer::new(config).unwrap();
            for text in corpus {
                trainer.add(text);
            }
            trainer.finish().unwrap().0.merges().to_vec()
        };
        // Pieces: "ab" twice (pair ab: 2) and "cd" once (pair cd: 1).
        assert_eq!(run(2), [(97, 98)]);
        assert_eq!(run(1), [(97, 98), (99, 100)]);
        assert!(run(3).is_empty());
    }

    #[test]
    fn special_tokens_take_the_last_ids_and_count_toward_the_vocabulary() {
        let mut config = TrainerConfig::new(256 + 2 + 2);
        config.special_tokens = vec!["<|a|>".to_string(), "<|b|>".to_string()];
        let mut trainer = Trainer::new(config).unwrap();
        trainer.add_counted(b"xyxy", 4);
        trainer.add_counted(b"xy", 4);
        let (tokenizer, stats) = trainer.finish().unwrap();
        assert_eq!(stats.requested_merges, 2);
        assert_eq!(tokenizer.merges().len(), 2);
        assert_eq!(tokenizer.vocab_size(), 260);
        assert_eq!(tokenizer.special_id("<|a|>"), Some(258));
        assert_eq!(tokenizer.special_id("<|b|>"), Some(259));
    }

    #[test]
    fn bad_configurations_are_rejected() {
        assert!(Trainer::new(TrainerConfig::new(255)).is_err());
        let mut config = TrainerConfig::new(257);
        config.special_tokens = vec!["<a>".to_string(), "<b>".to_string()];
        assert!(Trainer::new(config).is_err(), "no room for two specials");
        let mut config = TrainerConfig::new(300);
        config.min_frequency = 0;
        assert!(Trainer::new(config).is_err());
        let mut config = TrainerConfig::new(300);
        config.special_tokens = vec!["bad\nname".to_string()];
        assert!(Trainer::new(config).is_err());
        assert!(Trainer::new(TrainerConfig::new(256)).is_ok());
    }

    #[test]
    fn add_ignores_empty_input_and_zero_counts() {
        let mut trainer = Trainer::new(TrainerConfig::new(300)).unwrap();
        trainer.add(b"");
        trainer.add_counted(b"", 5);
        trainer.add_counted(b"ab", 0);
        let (_, stats) = trainer.finish().unwrap();
        assert_eq!((stats.unique_pieces, stats.pieces_seen, stats.bytes_seen), (0, 0, 0));
    }

    const SAMPLE: &str = "fn main() {\n    let mut total = 0;\n    for i in 0..100 {\n        total += i * 2;\n    }\n    println!(\"total = {}\", total);\n}\n\n$ cargo build --release\n\x1b[1;32m   Compiling\x1b[0m midman v0.0.1\n\x1b[1;32m    Finished\x1b[0m release profile in 3.21s\n$ git diff\n-    let x = 1;\n+    let x = 2;\n";

    fn train_on(text: &str, vocab: usize) -> (Tokenizer, TrainStats) {
        let mut config = TrainerConfig::new(vocab);
        config.min_frequency = 1;
        let mut trainer = Trainer::new(config).unwrap();
        for _ in 0..3 {
            trainer.add(text.as_bytes());
        }
        trainer.finish().unwrap()
    }

    #[test]
    fn merge_counts_never_increase_and_every_merge_is_used() {
        let (tokenizer, stats) = train_on(SAMPLE, 256 + 150);
        assert_eq!(stats.merges, tokenizer.merges().len());
        assert!(stats.merge_counts.windows(2).all(|w| w[0] >= w[1]), "{:?}", stats.merge_counts);
        assert!(stats.merge_counts.iter().all(|&c| c >= 1));
    }

    #[test]
    fn training_text_round_trips_and_more_merges_never_give_more_tokens() {
        let (full, _) = train_on(SAMPLE, 256 + 150);
        let ids = full.encode(SAMPLE, SpecialMode::AsText);
        assert_eq!(full.decode(&ids).unwrap(), SAMPLE);
        assert!(ids.len() < SAMPLE.len() / 2, "{} tokens for {} bytes", ids.len(), SAMPLE.len());

        let mut previous = usize::MAX;
        for k in [0, 10, 40, 80, 150] {
            let merges = full.merges()[..k.min(full.merges().len())].to_vec();
            let prefix = Tokenizer::new(*full.pretokenizer(), merges, Vec::new()).unwrap();
            let n = prefix.encode(SAMPLE, SpecialMode::AsText).len();
            assert!(n <= previous, "{k} merges gave {n} tokens after {previous}");
            previous = n;
        }
    }

    #[test]
    fn the_encoder_reproduces_the_segmentation_the_trainer_ended_with() {
        // The trainer applies each merge to every word; the encoder must land on the
        // same tokens for every distinct piece.
        let config = PreTokenizerConfig::cl100k_like();
        let mut counts: HashMap<Vec<u8>, i64> = HashMap::new();
        for _ in 0..3 {
            for piece in config.pieces(SAMPLE.as_bytes()) {
                *counts.entry(piece.to_vec()).or_insert(0) += 1;
            }
        }
        let words: Vec<Word> = counts
            .iter()
            .map(|(bytes, &count)| Word {
                symbols: bytes.iter().map(|&b| u32::from(b)).collect(),
                count,
            })
            .collect();
        let originals: Vec<Vec<u8>> = counts.keys().cloned().collect();
        // `counts.iter()` and `counts.keys()` walk the map in the same order.
        let learned = learn(words, 120, 1);
        let tokenizer = Tokenizer::new(config, learned.merges.clone(), Vec::new()).unwrap();
        for (word, original) in learned.words.iter().zip(&originals) {
            let encoded = tokenizer.encode_bytes(original, SpecialMode::AsText);
            assert_eq!(encoded, word.symbols, "{}", String::from_utf8_lossy(original));
        }
    }

    #[test]
    fn long_runs_of_one_symbol_are_counted_by_position_but_merge_without_overlap() {
        let train = |merges: usize| {
            let mut config = TrainerConfig::new(256 + merges);
            config.min_frequency = 1;
            let mut trainer = Trainer::new(config).unwrap();
            trainer.add_counted(&[b'a'; 5], 1);
            trainer.finish().unwrap()
        };
        // aaaaa has the pair (a,a) at four positions, and merging leaves (aa)(aa)a.
        let (one, stats) = train(1);
        assert_eq!(one.merges(), [(97, 97)]);
        assert_eq!(stats.merge_counts, [4]);
        assert_eq!(one.encode("aaaaa", SpecialMode::AsText), [256, 256, 97]);
        // Then (aa,a) and (aa,aa) tie at one position each, and the smaller pair
        // (aa,a) goes first, leaving aa (aa a); the last merge joins what is left.
        let (three, stats) = train(3);
        assert_eq!(three.merges(), [(97, 97), (256, 97), (256, 257)]);
        assert_eq!(stats.merge_counts, [4, 1, 1]);
        assert_eq!(three.encode("aaaaa", SpecialMode::AsText), [258]);
    }

    #[test]
    fn a_large_random_corpus_trains_consistently_with_the_encoder() {
        let mut rng = Rng::seed_from_u64(11);
        let words = [
            "let", "mut", "fn", "pub", "struct", "impl", "match", "self", "Vec", "usize", "string",
        ];
        let mut text = String::new();
        for _ in 0..4000 {
            text.push_str(words[rng.below(words.len())]);
            text.push(match rng.below(5) {
                0 => '\n',
                1 => '(',
                2 => ';',
                _ => ' ',
            });
            if rng.below(8) == 0 {
                text.push_str(&rng.below(1000).to_string());
            }
        }
        let (tokenizer, stats) = train_on(&text, 256 + 200);
        assert_eq!(stats.merges, 200);
        let ids = tokenizer.encode(&text, SpecialMode::AsText);
        assert_eq!(tokenizer.decode(&ids).unwrap(), text);
        assert!(ids.len() * 2 < text.len(), "{} tokens for {} bytes", ids.len(), text.len());
    }
}
