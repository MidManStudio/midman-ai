// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-data.md, section "batch.rs"
// ============================================================================
//! Training sequences cut from shards, and batches of them.

use midman_foundation::{Error, Result, Rng};

use crate::shard::TokenShard;

/// The training sequences in a set of shards.
///
/// A sequence is a window of `seq_len + 1` consecutive tokens of one shard: the first
/// `seq_len` are the inputs and the last `seq_len` are the targets, so each target is the
/// token that follows its input. Windows start every `seq_len` tokens, so neighbors share one
/// token, every token is a target exactly once, and no window crosses from one shard to the
/// next. Where a shard ends with fewer than `seq_len + 1` tokens left, those tokens are not
/// used. Documents are not kept apart: a window can hold the end of one document and the
/// start of the next, separated by the end-of-text token.
///
/// ```
/// use midman_data::{Sequences, TokenShard};
///
/// let shard = TokenShard::new(100, 99, (0..11).collect()).unwrap();
/// let sequences = Sequences::new(vec![shard], 4).unwrap();
/// assert_eq!(sequences.len(), 2);
/// assert_eq!(sequences.window(0).unwrap(), [0, 1, 2, 3, 4]);
/// assert_eq!(sequences.window(1).unwrap(), [4, 5, 6, 7, 8]);
/// ```
#[derive(Debug, Clone)]
pub struct Sequences {
    shards: Vec<TokenShard>,
    seq_len: usize,
    /// For each shard, the number of sequences in it and all the shards before it.
    ends: Vec<usize>,
}

impl Sequences {
    /// The sequences of `seq_len` tokens in `shards`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `seq_len` is 0 or if the shards do not all have
    /// the same vocabulary size and end-of-text id.
    pub fn new(shards: Vec<TokenShard>, seq_len: usize) -> Result<Sequences> {
        if seq_len == 0 {
            return Err(Error::invalid("a sequence must have at least one token"));
        }
        if let Some(first) = shards.first() {
            for (n, shard) in shards.iter().enumerate() {
                if (shard.vocab_size(), shard.eos()) != (first.vocab_size(), first.eos()) {
                    return Err(Error::invalid(format!(
                        "shard {n} has vocabulary size {} and end-of-text {}, but shard 0 has {} and {}",
                        shard.vocab_size(),
                        shard.eos(),
                        first.vocab_size(),
                        first.eos()
                    )));
                }
            }
        }
        let mut ends = Vec::with_capacity(shards.len());
        let mut total = 0usize;
        for shard in &shards {
            if shard.len() > seq_len {
                total += (shard.len() - 1) / seq_len;
            }
            ends.push(total);
        }
        Ok(Sequences { shards, seq_len, ends })
    }

    /// The number of input tokens in a sequence.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// The number of sequences.
    pub fn len(&self) -> usize {
        self.ends.last().copied().unwrap_or(0)
    }

    /// Whether there are no sequences.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The vocabulary size of the shards, or `None` if there are no shards.
    pub fn vocab_size(&self) -> Option<u32> {
        self.shards.first().map(TokenShard::vocab_size)
    }

    /// The window of `seq_len + 1` tokens of sequence `index`, or `None` if there is no such
    /// sequence.
    pub fn window(&self, index: usize) -> Option<&[u32]> {
        if index >= self.len() {
            return None;
        }
        let shard = self.ends.partition_point(|&end| end <= index);
        let before = if shard == 0 { 0 } else { self.ends[shard - 1] };
        let start = (index - before) * self.seq_len;
        Some(&self.shards[shard].tokens()[start..start + self.seq_len + 1])
    }
}

/// One batch for `MidManModel::loss`: `inputs` and `targets` are `batch * seq` token ids each,
/// row after row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    /// The input ids, `batch` rows of `seq`.
    pub inputs: Vec<usize>,
    /// The target ids: row `r`, position `t` is the token after `inputs` row `r`, position `t`.
    pub targets: Vec<usize>,
    /// The number of rows.
    pub batch: usize,
    /// The number of tokens in a row.
    pub seq: usize,
    /// The pass over the data this batch belongs to, from 0.
    pub epoch: u64,
    /// The position of this batch in its pass, from 0.
    pub index: usize,
}

/// Cuts [`Sequences`] into batches, endlessly, in an order that is fixed by a seed.
///
/// ```
/// use midman_data::{Batcher, Sequences, TokenShard};
///
/// let shard = TokenShard::new(100, 99, (0..41).map(|t| t % 90).collect()).unwrap();
/// let sequences = Sequences::new(vec![shard], 4).unwrap(); // 10 sequences
/// let mut batcher = Batcher::new(sequences, 3, 7).unwrap();
/// assert_eq!(batcher.batches_per_epoch(), 3);
/// let batch = batcher.next_batch();
/// assert_eq!((batch.batch, batch.seq, batch.inputs.len()), (3, 4, 12));
/// ```
///
/// Each pass over the data (an epoch) visits the sequences in a fresh random order, taking
/// them `batch` at a time; the few sequences left over when the count is not a multiple of
/// the batch size are skipped for that pass, and a different few are skipped in the next. The
/// order of a pass depends only on the seed and the pass number, so a run that stopped can be
/// continued exactly: note [`Batcher::epoch`] and [`Batcher::position`], and give them to
/// [`Batcher::seek`] on a batcher made with the same seed and data.
#[derive(Debug)]
pub struct Batcher {
    data: Sequences,
    batch_size: usize,
    seed: u64,
    epoch: u64,
    /// Batches of the current epoch already handed out.
    next: usize,
    order: Vec<usize>,
}

impl Batcher {
    /// A batcher over `data` with `batch_size` sequences per batch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `batch_size` is 0 or there are fewer sequences
    /// than one batch.
    pub fn new(data: Sequences, batch_size: usize, seed: u64) -> Result<Batcher> {
        if batch_size == 0 {
            return Err(Error::invalid("a batch must have at least one sequence"));
        }
        if data.len() < batch_size {
            return Err(Error::invalid(format!(
                "there are {} sequences, fewer than one batch of {batch_size}",
                data.len()
            )));
        }
        let order = permutation(data.len(), seed, 0);
        Ok(Batcher { data, batch_size, seed, epoch: 0, next: 0, order })
    }

    /// The number of batches in one pass over the data.
    pub fn batches_per_epoch(&self) -> usize {
        self.data.len() / self.batch_size
    }

    /// The pass the next batch belongs to, unless the current one is used up (then the next
    /// batch starts the following pass).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// How many batches of [`Batcher::epoch`] have been handed out.
    pub fn position(&self) -> usize {
        self.next
    }

    /// Jumps to `epoch` with `batches_done` of its batches already handed out, so that the
    /// next batch is the one a continuous run would have produced next.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `batches_done` is more than the batches in a pass.
    pub fn seek(&mut self, epoch: u64, batches_done: usize) -> Result<()> {
        if batches_done > self.batches_per_epoch() {
            return Err(Error::invalid(format!(
                "{batches_done} batches done is more than the {} in a pass",
                self.batches_per_epoch()
            )));
        }
        if epoch != self.epoch {
            self.order = permutation(self.data.len(), self.seed, epoch);
        }
        self.epoch = epoch;
        self.next = batches_done;
        Ok(())
    }

    /// The next batch. There is always one: when a pass is used up, the next begins.
    pub fn next_batch(&mut self) -> Batch {
        if self.next == self.batches_per_epoch() {
            self.epoch += 1;
            self.next = 0;
            self.order = permutation(self.data.len(), self.seed, self.epoch);
        }
        let seq = self.data.seq_len();
        let rows = &self.order[self.next * self.batch_size..(self.next + 1) * self.batch_size];
        let mut inputs = Vec::with_capacity(self.batch_size * seq);
        let mut targets = Vec::with_capacity(self.batch_size * seq);
        for &sequence in rows {
            // Every index in `order` is below `data.len()`, so the window exists.
            if let Some(window) = self.data.window(sequence) {
                inputs.extend(window[..seq].iter().map(|&t| t as usize));
                targets.extend(window[1..].iter().map(|&t| t as usize));
            }
        }
        let batch = Batch {
            inputs,
            targets,
            batch: self.batch_size,
            seq,
            epoch: self.epoch,
            index: self.next,
        };
        self.next += 1;
        batch
    }
}

/// The order of the `n` sequences in pass `epoch`, from `seed` alone.
fn permutation(n: usize, seed: u64, epoch: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    Rng::seed_from_u64(seed).fork(epoch).shuffle(&mut order);
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn stream(n: u32, vocab: u32) -> TokenShard {
        TokenShard::new(vocab, vocab - 1, (0..n).map(|t| t % (vocab - 1)).collect()).unwrap()
    }

    fn sequences(tokens: u32, seq_len: usize) -> Sequences {
        Sequences::new(vec![stream(tokens, 1000)], seq_len).unwrap()
    }

    #[test]
    fn windows_start_every_seq_len_tokens_and_overlap_by_one() {
        let s = sequences(21, 4);
        assert_eq!(s.len(), 5);
        for i in 0..5 {
            let start = (i * 4) as u32;
            assert_eq!(s.window(i).unwrap(), (start..start + 5).collect::<Vec<_>>());
        }
        assert!(s.window(5).is_none());
        assert_eq!(s.seq_len(), 4);
        assert_eq!(s.vocab_size(), Some(1000));
    }

    #[test]
    fn every_token_is_a_target_exactly_once_and_the_remainder_is_dropped() {
        // 23 tokens, seq_len 4: windows at 0, 4, 8, 12, 16 use tokens 0..=20, so 21 and 22
        // are dropped.
        let s = sequences(23, 4);
        assert_eq!(s.len(), 5);
        let mut targets = Vec::new();
        for i in 0..s.len() {
            targets.extend_from_slice(&s.window(i).unwrap()[1..]);
        }
        assert_eq!(targets, (1..=20).collect::<Vec<u32>>());
    }

    #[test]
    fn the_count_is_exact_at_the_edges() {
        for (tokens, expected) in [(0, 0), (1, 0), (4, 0), (5, 1), (8, 1), (9, 2), (13, 3)] {
            assert_eq!(sequences(tokens, 4).len(), expected, "{tokens} tokens");
        }
        assert!(sequences(4, 4).is_empty());
        assert_eq!(sequences(5, 4).window(0).unwrap().len(), 5);
        // seq_len 1: every token but the first is a target.
        assert_eq!(sequences(10, 1).len(), 9);
    }

    #[test]
    fn windows_never_cross_from_one_shard_to_the_next() {
        let a = TokenShard::new(1000, 999, (0..10).collect()).unwrap();
        let b = TokenShard::new(1000, 999, (100..109).collect()).unwrap();
        let c = TokenShard::new(1000, 999, (200..203).collect()).unwrap();
        let d = TokenShard::new(1000, 999, (300..306).collect()).unwrap();
        let s = Sequences::new(vec![a, b, c, d], 4).unwrap();
        // 10 tokens: 2 windows; 9: 2; 3: 0; 6: 1.
        assert_eq!(s.len(), 5);
        let windows: Vec<Vec<u32>> = (0..5).map(|i| s.window(i).unwrap().to_vec()).collect();
        assert_eq!(
            windows,
            [
                vec![0, 1, 2, 3, 4],
                vec![4, 5, 6, 7, 8],
                vec![100, 101, 102, 103, 104],
                vec![104, 105, 106, 107, 108],
                vec![300, 301, 302, 303, 304],
            ]
        );
    }

    #[test]
    fn no_shards_and_bad_arguments() {
        let none = Sequences::new(Vec::new(), 4).unwrap();
        assert!(none.is_empty() && none.vocab_size().is_none() && none.window(0).is_none());
        assert!(Sequences::new(vec![stream(10, 100)], 0).is_err());
        let other_vocab = TokenShard::new(200, 199, vec![1; 10]).unwrap();
        let other_eos = TokenShard::new(100, 98, vec![1; 10]).unwrap();
        let err = Sequences::new(vec![stream(10, 100), other_vocab], 4).unwrap_err();
        assert!(err.to_string().contains("shard 1"), "{err}");
        assert!(Sequences::new(vec![stream(10, 100), other_eos], 4).is_err());
    }

    #[test]
    fn batches_hold_inputs_and_the_next_tokens_as_targets() {
        let mut batcher = Batcher::new(sequences(101, 5), 4, 1).unwrap();
        for _ in 0..12 {
            let batch = batcher.next_batch();
            assert_eq!((batch.batch, batch.seq), (4, 5));
            assert_eq!((batch.inputs.len(), batch.targets.len()), (20, 20));
            for row in 0..4 {
                let inputs = &batch.inputs[row * 5..(row + 1) * 5];
                let targets = &batch.targets[row * 5..(row + 1) * 5];
                // The stream counts up by one, so each target is its input plus one.
                for (i, t) in inputs.iter().zip(targets) {
                    assert_eq!(*t, i + 1);
                }
                assert_eq!(&inputs[1..], &targets[..4]);
            }
        }
    }

    #[test]
    fn one_epoch_uses_each_sequence_at_most_once_and_skips_the_remainder() {
        let s = sequences(1 + 4 * 23, 4); // 23 sequences
        assert_eq!(s.len(), 23);
        let mut batcher = Batcher::new(s, 5, 9).unwrap();
        assert_eq!(batcher.batches_per_epoch(), 4);
        let mut seen = BTreeSet::new();
        for index in 0..4 {
            let batch = batcher.next_batch();
            assert_eq!((batch.epoch, batch.index), (0, index));
            for row in 0..5 {
                // Each sequence starts at a distinct multiple of 4 in this stream.
                seen.insert(batch.inputs[row * 4]);
            }
        }
        assert_eq!(seen.len(), 20, "20 distinct sequences, 3 skipped");
        let batch = batcher.next_batch();
        assert_eq!((batch.epoch, batch.index), (1, 0));
    }

    #[test]
    fn every_sequence_is_used_over_enough_epochs() {
        let mut batcher = Batcher::new(sequences(1 + 4 * 23, 4), 5, 3).unwrap();
        let mut seen = BTreeSet::new();
        for _ in 0..40 {
            let batch = batcher.next_batch();
            for row in 0..5 {
                seen.insert(batch.inputs[row * 4]);
            }
        }
        assert_eq!(seen.len(), 23);
    }

    #[test]
    fn the_order_depends_on_the_seed_and_the_epoch_alone() {
        let run = |seed: u64| {
            let mut b = Batcher::new(sequences(1 + 4 * 40, 4), 4, seed).unwrap();
            (0..25).map(|_| b.next_batch()).collect::<Vec<_>>()
        };
        assert_eq!(run(11), run(11));
        assert_ne!(run(11), run(12));
        // Epochs differ from each other.
        let batches = run(11);
        assert_eq!(batches[0].epoch, 0);
        assert_eq!(batches[10].epoch, 1);
        assert_ne!(batches[0].inputs, batches[10].inputs);
        assert_ne!(batches[0].inputs, batches[20].inputs);
    }

    #[test]
    fn seeking_continues_exactly_where_a_run_left_off() {
        let make = || Batcher::new(sequences(1 + 4 * 37, 4), 4, 5).unwrap();
        let mut straight = make();
        let all: Vec<Batch> = (0..40).map(|_| straight.next_batch()).collect();
        // Stop after k batches for every k, including the ends of epochs, and continue.
        for k in 0..=30 {
            let mut first = make();
            for _ in 0..k {
                first.next_batch();
            }
            let (epoch, position) = (first.epoch(), first.position());
            let mut second = make();
            second.seek(epoch, position).unwrap();
            for expected in &all[k..k + 8] {
                assert_eq!(&second.next_batch(), expected, "after {k} batches");
            }
        }
    }

    #[test]
    fn seek_checks_its_position_and_can_go_backwards() {
        let mut batcher = Batcher::new(sequences(1 + 4 * 12, 4), 4, 2).unwrap();
        assert_eq!(batcher.batches_per_epoch(), 3);
        assert!(batcher.seek(0, 4).is_err());
        assert!(batcher.seek(0, 3).is_ok());
        let first = batcher.next_batch();
        assert_eq!((first.epoch, first.index), (1, 0));
        batcher.seek(0, 0).unwrap();
        assert_eq!(batcher.next_batch().index, 0);
        batcher.seek(7, 2).unwrap();
        let batch = batcher.next_batch();
        assert_eq!((batch.epoch, batch.index), (7, 2));
        assert_eq!((batcher.epoch(), batcher.position()), (7, 3));
    }

    #[test]
    fn construction_checks_its_arguments() {
        assert!(Batcher::new(sequences(1 + 4 * 5, 4), 0, 1).is_err());
        let err = Batcher::new(sequences(1 + 4 * 5, 4), 6, 1).unwrap_err();
        assert!(err.to_string().contains("5 sequences"), "{err}");
        assert!(Batcher::new(sequences(1 + 4 * 5, 4), 5, 1).is_ok());
        assert!(Batcher::new(Sequences::new(Vec::new(), 4).unwrap(), 1, 1).is_err());
    }

    #[test]
    fn a_batch_of_one_sequence_per_pass_works() {
        let mut batcher = Batcher::new(sequences(1 + 4, 4), 1, 1).unwrap();
        for epoch in 0..3 {
            let batch = batcher.next_batch();
            assert_eq!((batch.epoch, batch.index), (epoch, 0));
        }
    }
}
